import argparse
def get_args(jupyter=False):
    parser = argparse.ArgumentParser()
    # Distributed and accelerate setting
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=2,
                        help='only used in supervise_cross_ddp.py, auto recognized most of time')
    parser.add_argument('--use_amp', type=str2bool, default=True,
                        help='whether to use automatic precision for training')
    parser.add_argument('--k', type=int, default=0, help='k_th fold, ineffective for cross training')

    # Training setting
    parser.add_argument('--monitor', type=str, default='accuracy', help='accuracy| f1 | rscore')
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--expid', type=str, default='test', help='experiment name')
    parser.add_argument('--model', type=str, default='densenet121',
                        help='densenet[121|161]|resenet|se_resnet|resnext|wideresnet[50|101|152]|efficientnet-b[0-7]')
    parser.add_argument('--epochs', type=int, default=30, help='total epochs for supervised training')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for labeled set')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--sampler', type=str, default='balanced', help='balanced|standard|sqrt')
    parser.add_argument('--log_dir', type=str, default='outputs')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='ckpt for supervise learning or teacher network')

    # Optimizer setting 
    parser.add_argument('--op_type', type=str, default='AdamW', help='or SGD')
    parser.add_argument('--lr', type=float, default=1.0e-3)
    parser.add_argument('--beta0', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    # Scheduler setting
    parser.add_argument('--scheduler_type', type=str, default='OneCycle', 
                        help="or Cos for CosineAnnealingLR, else MultiStepLR")                    
    parser.add_argument('--total_steps', type=int, default=600, help="for OneCycle")
    parser.add_argument('--T_max', type=int, default=1000, help="for CosineAnnealingLR")
    parser.add_argument('--milestones', nargs='+', type=int, default=[100], help='for MultiStepLR')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='for MultiStepLR')
    
    # Param for distillation
    parser.add_argument('--distill_ckpt', type=str, 
                        default='/home/ljc/keratitis/byol_save/byol_1000.pth',
                        help='byol backbone for student network')
    parser.add_argument('--distill_bs', type=int, default=128, help='batch size for unlabeled set')
    parser.add_argument('--distill_epochs', type=int, default=30, help='total epochs for semisl training in each iter')
    parser.add_argument('--distill_iter', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.9, help='filter out pseudo labels with threshold')
    parser.add_argument('--pseudo_form', type=str, default='hard',
                        help='soft|hard|kl, soft label or hard label for pseudo label')
    parser.add_argument('--unl_aug', type=str2bool, default=False,
                        help='whether to use augmentaion in unlabeled set')
    parser.add_argument('--strategy', type=str, default='consistent',
                        help='single|consistent, pseudo label filter strategy')
    parser.add_argument('--boundary', nargs='+', type=float, default=[0.1, 0.3, 0.2, 0.4],
                        help='param for consitent only, min_l, max_l, min_u, max_u')
    parser.add_argument('--alpha', type=float, default=1.0, help='self-training loss weight')

    if jupyter == False:
        args = parser.parse_args()
    else:  
        parser.add_argument("--verbosity", help="increase output verbosity")    
        args = parser.parse_args(args=[])
    return args

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


