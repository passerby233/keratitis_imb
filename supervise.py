import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from sklearn import metrics
import json
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

from args import get_args
from utils import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s: %(message)s')

def train(model, dataloader, optimizer, scheduler, use_amp):
    model.train()
    scaler = GradScaler()
    device = next(model.parameters()).device
    loss_list, y_pred, y_true = [], [], []
    for step, data in enumerate(dataloader):
        image, target = move_to_device(data, device)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                logit = model(image)
            loss = F.cross_entropy(logit, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logit = model(image) # B x C
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        loss_value = loss.item()
        loss_list.append(loss_value)
        conf = F.softmax(logit, 1, dtype=torch.float32).cpu().detach().numpy()
        if step == 0:
            y_score = conf
        else:
            y_score = np.concatenate((y_score, conf), axis=0)
        y_pred.extend(logit.argmax(-1).cpu().tolist())
        y_true.extend(target.cpu().tolist())
        print(f"Step {step}: Loss: {loss_value}")

    # Record
    train_loss = np.mean(loss_list)
    report = metrics.classification_report(y_true=y_true, y_pred=y_pred, digits=3,
                                           target_names=dataloader.dataset.classes,  
                                           zero_division=0, output_dict=False)
    report_dict = metrics.classification_report(y_true=y_true, y_pred=y_pred, digits=3,
                                                target_names=dataloader.dataset.classes,  
                                                zero_division=0, output_dict=True)
    # report auc                                            
    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score, multi_class='ovo')
    report_dict['auc'] = auc
    # extra r_score for selecting models
    recalls = [report_dict[key]['recall'] for key in dataloader.dataset.classes]
    rscore = np.mean(recalls) / (np.max(recalls) - np.min(recalls)) # mean/(max-min)
    bias = np.std(recalls)
    # add to record
    report_dict['rscore'] = rscore
    report_dict['bias'] = bias
    report_dict['f1'] = report_dict['macro avg']['f1-score']
    return report, report_dict, train_loss

def test(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    y_pred, y_true = [], []
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            images, target = data
            images = images.to(device)
            logit = model(images).cpu()
            y_pred.extend(logit.argmax(-1).tolist())
            y_true.extend(target.tolist())
            conf = F.softmax(logit, 1, dtype=torch.float32).cpu().detach().numpy()
            if step == 0:
                y_score = conf
            else:
                y_score = np.concatenate((y_score, conf), axis=0)

    report = metrics.classification_report(y_true=y_true, y_pred=y_pred, digits=3,
                                           target_names=dataloader.dataset.classes,  
                                           zero_division=0, output_dict=False)
    report_dict = metrics.classification_report(y_true=y_true, y_pred=y_pred, digits=3,
                                                target_names=dataloader.dataset.classes,  
                                                zero_division=0, output_dict=True)
    # report auc                                            
    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score, multi_class='ovo')
    report_dict['auc'] = auc
    # extra r_score for selecting models
    recalls = [report_dict[key]['recall'] for key in dataloader.dataset.classes]
    rscore = np.mean(recalls) / (np.max(recalls) - np.min(recalls)) # mean/(max-min)
    bias = np.std(recalls)
    # add to record
    report_dict['rscore'] = rscore
    report_dict['bias'] = bias
    report_dict['f1'] = report_dict['macro avg']['f1-score']
    return report, report_dict

def main():
    seed_torch()
    args = get_args()

    report_list = []
    for k in range(5):
        args.k = k
        exp_dir, save_dir, writer = get_utils_cross(args)
        save_path = os.path.join(save_dir, f'{args.model}.pth')
        report_path = os.path.join(save_dir, f'{args.model}_{k}.json')

        # Get data, model, optimizer, scheduler
        dataset, loader = get_loader(args)
        args.total_steps = args.epochs * len(loader['train'])
        logging.info(f"Length of Train, Val dataset : {len(dataset['train']), len(dataset['val'])}")
        logging.info(f"Total Steps : {args.total_steps}")
    
        model = get_model(args.model, len(dataset['val'].classes)).cuda()
        load_backbone_from_args(model, args)
        model = torch.nn.DataParallel(model)
        
        optimizer = get_optimizer(model.parameters(), args)
        scheduler = get_scheduler(optimizer, args)
        if args.use_amp:
            logging.info("AMP activated")

        # Train and Test Iteration
        max_metric = 0
        best_report, best_report_dict = None, None
        logging.info(f'Start training, k={k}')
        for epoch in range(args.epochs):
            for i, mode in enumerate(['train', 'val']):
                if mode == 'train':
                    report, report_dict, loss = train(model, loader['train'], optimizer, scheduler, args.use_amp)
                    logging.info(f"{args.expid}, Epoch {epoch}:") 
                    logging.info(f"{mode.capitalize():5} Acc:{report_dict['accuracy']:.3f} " +\
                                 f"{mode.capitalize():5} F1: {report_dict['macro avg']['f1-score']:.3f} " +\
                                 f"{mode.capitalize():5} AUC: {report_dict['auc']:.3f} " +\
                                 f"{mode.capitalize():5} Loss: {loss}")
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                    writer.add_scalar(f'{i}_{mode}/3_loss', loss, epoch)
                else:
                    report, report_dict= test(model, loader['val'])
                    logging.info(f"{mode.capitalize():5} Acc:{report_dict['accuracy']:.3f} " +\
                                 f"{mode.capitalize():5} F1: {report_dict['macro avg']['f1-score']:.3f} " +\
                                 f"{mode.capitalize():5} AUC: {report_dict['auc']:.3f} ") 
                    if report_dict[args.monitor] > max_metric:    
                        max_metric, best_report, best_report_dict = report_dict[args.monitor], report, report_dict
                        logging.info(f"Saving model, {args.model} got new max_{args.monitor}: {max_metric:3}")
                        print(report)
                        # Try to save best on val or save at end of training
                        # torch.save(model.state_dict(), save_path)
            
                # log to tensorboard    
                for num, metric_name in enumerate(['accuracy', 'auc', 'f1', 'rscore', 'bias']):
                    writer.add_scalar(f'{i}_{mode}/{num}_{metric_name}', report_dict[metric_name], epoch) 

        logging.info("Last epoch on validation:")
        print(report)                             
        # Reloading Best Model on Validation and Test on Test Set
        if hasattr(model, "module"):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
        best_report, best_report_dict = test(model, loader['test'])
        print('Report on Test Set:\n' + best_report)
        with open(report_path ,'w') as f:
            json.dump(best_report_dict, f)
        report_list.append(best_report_dict)

    # get average report of 5_cross, save to path
    avg_report_path = os.path.join(exp_dir, f'{args.expid}_cross.json')
    with open(avg_report_path ,'w') as f:
        json.dump(merge_report(report_list), f)

if __name__ == '__main__':
    import warnings
    # to suppress annoying warnings from PIL
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    main()