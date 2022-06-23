import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import numpy as np
import torch, json, shutil
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from dataset import KeratitisLabeled, KeratitisUnlabeled, test_aug, semisl_aug
from args import get_args
from utils import get_model, get_optimizer, get_utils_cross, load_backbone, merge_report
from supervise import seed_torch
from ssl_algorithm import get_pseudo, ssl_train
import ssl_algorithm
from supervise import test


import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s: %(message)s')

def main():
    seed_torch()

    # Get basic utils
    args = get_args()
    report_list = []
    for k in range(5):
        args.k = k
        exp_dir, save_dir, writer = get_utils_cross(args)
        tmp_save_path = os.path.join(save_dir, f'tmp_{k}.pth')
        model_save_path = os.path.join(save_dir, f'{args.model}_{k}.pth')
        report_save_path = os.path.join(save_dir, f'{args.expid}_{k}.json')
        print(tmp_save_path, model_save_path, report_save_path)

        # Create dataset of labeled, unlabeled, test
        unlabeled_dataset = KeratitisUnlabeled(transform=test_aug)
        unlabeled_augset = KeratitisUnlabeled(transform=semisl_aug)
        labeled_trainset = KeratitisLabeled(mode='train', k=k)
        labeled_valset = KeratitisLabeled(mode='val', k=k)
        labeled_testset = KeratitisLabeled(mode='test', k=k)
        test_loader = DataLoader(labeled_testset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        # Create teacher model from pretrained model
        t_model_path = os.path.join(args.ckpt, f'fold_{k}/densenet121.pth')
        logging.info(f'Fold{k}: Loading pretrained teacher model from:\n{t_model_path}')
        t_model = get_model(args.model, len(labeled_trainset.classes)).cuda()
        t_model.load_state_dict(torch.load(t_model_path))
        t_model = torch.nn.DataParallel(t_model)

        max_metric, best_report, best_report_dict = 0,  None, None
        pseudo_list = []
        ema_list = None
        for iter_step in range(args.distill_iter):
            # Get pseudo, filter for unlabeled set
            logging.info(f'Fold {k}: Inferencing on unlabeled data to get pseudo labels')
            pseudo_save_path = os.path.join(save_dir, f'pseudo_{iter_step}.npy')
            if args.debug and os.path.exists(pseudo_save_path) and iter_step==0:
                single_step_pseudo = np.load(pseudo_save_path)
            else:
                single_step_pseudo = get_pseudo(t_model, unlabeled_dataset)
                if args.debug:
                    np.save(pseudo_save_path, single_step_pseudo)
            pseudo_list.append(single_step_pseudo)
            filter_strategy = getattr(ssl_algorithm, args.strategy)
            pseudo, id_pool, class_length, pseudo_std, ema_list = filter_strategy(pseudo_list, args.threshold,ema_list)
            unlabled_set = unlabeled_augset if args.unl_aug else unlabeled_dataset
            selected_unlset = Subset(unlabled_set, id_pool)
            logging.info(f'Fold {k}: Iter {iter_step} selected unlabeled data: {class_length}')

            # Consistent strategy compute extra std for filter
            if args.strategy == 'consistent':
                pseudo_std_save_path = os.path.join(save_dir, f'pseudo_std_{iter_step}.npy')
                std = pseudo_std[np.arange(pseudo_std.shape[0]), np.argmax(single_step_pseudo, -1)]
                if args.debug:
                    np.save(pseudo_std_save_path, std)

            # Remove teacher mode and create a new student model'
            del t_model
            torch.cuda.empty_cache()
            s_model = get_model(args.model, len(labeled_trainset.classes)).cuda()
            if os.path.exists(args.distill_ckpt):
                logging.info(f'Fold {k}, Iter {iter_step}: Init student model with {args.distill_ckpt}')
                load_backbone(s_model, args.distill_ckpt)
            s_model = torch.nn.DataParallel(s_model)
            optimizer = get_optimizer(s_model.parameters(), args)

            # Distill t_model to s_model with both labeled data & unlabeled data
            report, report_dict = ssl_train(s_model, labeled_trainset, labeled_valset, args.monitor,
                                        selected_unlset, pseudo, class_length, pseudo_std,
                                        optimizer, iter_step, args, tmp_save_path, alpha=args.alpha)
            for num, metric_name in enumerate(['accuracy', 'auc', 'f1', 'rscore', 'bias']):
                writer.add_scalar(f'1_val/{num}_{metric_name}', report_dict[metric_name])  
            logging.info("Last epoch on validation:")
            print(report)

            # Best Iter save to model_save_path
            # Best epoch in each iter save to tmp_save_path
            if report_dict[args.monitor] > max_metric:    
                max_metric  = report_dict[args.monitor] 
                best_report, best_report_dict = report, report_dict  
                shutil.copyfile(tmp_save_path, model_save_path)

            # Take s_model as new t_model for next iter, del optimizer 
            t_model = s_model
            del optimizer
        
        s_model.module.load_state_dict(torch.load(model_save_path))
        report, report_dict = test(s_model, test_loader)
        with open(report_save_path ,'w') as f:
            json.dump(best_report_dict, f)
        # Display the best result
        print('Report on Test Set:\n' + best_report)
        report_list.append(best_report_dict)

    avg_report_path = os.path.join(exp_dir, f'{args.expid}_cross.json')
    with open(avg_report_path ,'w') as f:
        json.dump(merge_report(report_list), f)


if __name__ == '__main__':
    import warnings
    # to suppress annoying warnings from PIL
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    main()
