import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from dataset import BalancedSampler
from supervise import test
from utils import save_model

from tqdm import tqdm
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s: %(message)s')


def get_pseudo(t_model, unlabeled_dataset):
    """
    params:
        t_model: teacher model
        unlabeled_dataset: unlabeled dataset
    return:
        pseudo_label: ndarray[N, C], N=len(dataloader), C for num of class, dim C is output of softmax
    """
    t_model.eval()
    device = next(t_model.parameters()).device
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1400, shuffle=False,
                                  num_workers=32, pin_memory=True)                                  
    with torch.no_grad():
        for step, data in enumerate(tqdm(unlabeled_loader)):
            images, _ = data
            images = images.to(device)
            batch_pseudo = F.softmax(t_model(images), 1, dtype=torch.float32).cpu().detach().numpy()
            if step == 0:
                pseudo_label = batch_pseudo
            else:
                pseudo_label = np.concatenate((pseudo_label, batch_pseudo), axis=0)
    return pseudo_label


def filter_by_threshold(pseudo_label, threshold, num_per_class=-1):
    """
    Param:
        pseudo_label: numpy array, NxC, prediction of all unlabeled data
        threshold: samples with classficication confidence over threshold
        num_per_class: reserve topK samples for each class, off when -1
    return:
        A list of numpy array, each corresponds ot a class index
    """
    selected_index_list = []
    prob = np.max(pseudo_label, 1)
    logits = np.argmax(pseudo_label, 1)
    for i in range(pseudo_label.shape[1]):
        class_index = np.where((logits == i) & (prob > threshold))[0]
        if num_per_class > 0:
            topk_index_of_class_index = np.argsort(prob[class_index])[-num_per_class:]
            class_topk_index = class_index[topk_index_of_class_index]  # indexes of top prob in a caertain class
            if class_topk_index.shape[0] > 0:
                class_topk_index = np.pad(class_topk_index, (0, num_per_class - class_topk_index.shape[0]), 'wrap')
                selected_index_list.append(class_topk_index)
            else:
                logging.info(f'Class {i} get no unlabeled data !')
        else:
            selected_index_list.append(class_index)
    return selected_index_list


def filter_out(pseudo, threshold):
    selected_index_list = filter_by_threshold(pseudo, threshold)
    class_length = [class_index.shape[0] for class_index in selected_index_list]
    selected_index = np.array([], dtype=np.int32)
    for class_index in selected_index_list:
        selected_index = np.concatenate((selected_index, class_index), axis=0)
    return selected_index, class_length


def single(pseudo_list, threshold, ema_list):
    """
    Param:
        pseudo_list: A list of pseudo (np.array[NxC])
    return:
        pseudo: [SxC], selected pseudo
        selected_index: [S,], raw_index in [N,]
        class_length: length of each class, for sampler
    """
    pseudo = pseudo_list[-1]
    selected_index, class_length = filter_out(pseudo, threshold)
    return pseudo, selected_index, class_length, None, None


def ema(a,ema_list=None):
    try:
        layer, col, row = a.shape
    except:
        print("function EMA need array.shape like (*, *, *)")
        return
    if type(ema_list)==type(None):
        ema_list = np.empty(shape=(layer, col, row))
        ema_list[0] = np.stack(a[0])
    ema_step = np.empty(shape=(1,col,row))
    for i in range(col):
        for j in range(row):
            ema_step[0][i][j] = (2*a[-1][i][j] + (layer-1)*ema_list[-1][i][j]) / (layer+1)
    ema_list =np.append(ema_list, ema_step, axis=0)
    return ema_list


def consistent(pseudo_list, threshold, ema_list):
    group_pseudo = np.stack(pseudo_list)  # [LxNxC]
    ema_list = ema(group_pseudo, ema_list)
    pseudo_mean = ema_list[-1]
    pseudo_std = np.std(group_pseudo, axis=0)
    selected_index, class_length = filter_out(pseudo_mean, threshold)
    return pseudo_mean, selected_index, class_length, pseudo_std, ema_list


def ssl_train(s_model, labeled_trainset, labeled_valset, monitor,
              selected_unlset, all_pseudo, class_length, all_pseudo_std,
              optimizer, T_cur, args, tmp_save_path, alpha=1.0):
    device = next(s_model.parameters()).device
    scaler = GradScaler()
    val_loader = DataLoader(labeled_valset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
    if args.sampler == 'balanced':
        raw_lab_loader = DataLoader(labeled_trainset, batch_size=args.batch_size,
                                    sampler=BalancedSampler(labeled_trainset),
                                    num_workers=args.num_workers, pin_memory=True)
        raw_unl_loader = DataLoader(selected_unlset, batch_size=args.distill_bs,
                                    sampler=BalancedSampler(selected_unlset, class_length),
                                    num_workers=args.num_workers, pin_memory=True)
    else:
        raw_lab_loader = DataLoader(labeled_trainset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)
        raw_unl_loader = DataLoader(selected_unlset, batch_size=args.distill_bs, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)

    unl_loader = iter(raw_unl_loader)
    all_pseudo = torch.from_numpy(all_pseudo)
    if args.strategy == 'consistent':
        all_pseudo_std = torch.from_numpy(all_pseudo_std)
        min_l, max_l, min_u, max_u = args.boundary
        lower = T_cur / args.distill_iter * (max_l - min_l) + min_l
        upper = T_cur / args.distill_iter * (max_u - min_u) + min_u

    logging.info(f'Distilling Procedure, Iter {T_cur}:')
    max_metric, best_epoch, best_report, best_report_dict = 0, 0, None, None
    with tqdm(total=args.distill_epochs, desc='Distilling') as pbar:
        for epoch in range(args.distill_epochs):
            s_model.train()  # To ensure BN to be updated in each iter
            for lab_images, y_label in raw_lab_loader:
                try:
                    unl_images, indexes = next(unl_loader)
                except StopIteration:
                    unl_loader = iter(raw_unl_loader)
                    unl_images, indexes = next(unl_loader)

                # Foward
                with autocast():
                    y_ls = s_model(lab_images.to(device))  # Prediction on Labeled
                    y_label = y_label.to(device)  # Labeled Ground Truth
                    L_sup = F.cross_entropy(y_ls, y_label)

                    y_us = s_model(unl_images.to(device))  # Prediction on Unlabeled
                    pseudo_pred = torch.index_select(all_pseudo, 0, indexes).to(device)  # [Batch x Class]
                    if args.strategy == 'consistent':
                        prob, pseudo_label = pseudo_pred.max(-1)  # [Batch, ] max prob and class label
                        if args.pseudo_form == 'hard':
                            std = torch.index_select(all_pseudo_std, 0, indexes).to(device)  # [Batch x Class]
                            std_arg = std[
                                torch.arange(pseudo_label.shape[0]), pseudo_label]  # [Batch, ] std of max prob class
                            # Subsection function of beta
                            beta = torch.where(std_arg <= lower, torch.ones(1, device=device),
                                            torch.where(std_arg >= upper, torch.zeros(1, device=device),
                                                        (std_arg - upper) / (lower - upper)))
                            L_unl = F.cross_entropy(y_us, pseudo_pred.argmax(-1), reduction='none') @ beta / y_us.shape[0]
                        else:
                            L_unl = F.cross_entropy(y_us, pseudo_label, reduction='none') @ prob / y_us.shape[0]
                    elif args.pseudo_form == 'hard':
                        L_unl = F.cross_entropy(y_us, pseudo_pred.argmax(-1))
                    elif args.pseudo_form == 'soft':
                        prob, pseudo_label = pseudo_pred.max(-1)
                        L_unl = F.cross_entropy(y_us, pseudo_label, reduction='none') @ prob / y_us.shape[0]
                    elif args.pseudo_form == 'kl':
                        L_unl = F.kl_div(F.log_softmax(y_us, dim=-1), pseudo_pred, reduction='batchmean')
                    else:
                        raise Exception(f'Unrecognized mode {args.pseudo_form}')
                    loss = L_sup + alpha * L_unl
                loss_value, L_sup_value, L_unl_value = loss.item(), L_sup.item(), L_unl.item()
                # Optimize step
                optimizer.zero_grad()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            # tqdm logging
            pbar.set_postfix({'loss': f'{loss_value:.3f}',
                              'Sup': f'{L_sup_value:.3f}',
                              'Unl': f'{L_unl_value:.3f}'})
            pbar.update()

        # Test at last epoch on validation set
        report, report_dict = test(s_model, val_loader)  # Will call s_model.eval() inside  
        save_model(s_model, tmp_save_path)

    return report, report_dict

