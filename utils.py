import os, re, logging, random
import numpy as np
from glob import glob
from functools import reduce
from shutil import register_unpack_format

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import densenet121  # only for pretrained model
from torch.hub import load_state_dict_from_url

# Use local file for version compatibility
from model.densenet import DenseNet  
from model.resnet import ResNet, Bottleneck
import model.senet.se_resnet as se_resnet
from efficientnet_pytorch import EfficientNet

from dataset import KeratitisLabeled, BalancedSampler, SqrtSampler

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def get_model(model_name, num_classes):
    LAYER = {'resnet50':[3, 4, 6, 3],
         'resnet101':[3, 4, 23, 3],
         'resnet152':[3, 8, 36, 3],
         'resnext50':[3, 4, 6, 3],
         'resnext101':[3, 4, 23, 3],
         'wideresnet50':[3, 4, 6, 3],
         'wideresnet101':[3, 4, 23, 3],
         'densenet121':(6, 12, 24, 16),
         'densenet161':(6, 12, 36, 24),
         'densenet169':(6, 12, 32, 32),
         'densenet201':(6, 12, 48, 32)}
         
    if model_name[:6] == 'resnet':
        return ResNet(Bottleneck, LAYER[model_name], num_classes)
    elif model_name[:8] == 'densenet':
        growth_rate, num_init_features = (48, 96) if model_name== 'densenet161' else (32, 64)
        return DenseNet(growth_rate, LAYER[model_name], num_init_features, num_classes=num_classes)
    elif model_name[:7] == 'resnext':
        groups = 32
        width_per_group = 4 if model_name == 'resnext50' else 8
        return ResNet(Bottleneck, LAYER[model_name], num_classes, groups=groups, width_per_group=width_per_group)
    elif model_name[:4] == 'wide':
        width_per_group = 64 * 2
        return ResNet(Bottleneck, LAYER[model_name], num_classes, width_per_group=width_per_group)
    elif model_name[:9] == 'se_resnet':
        return getattr(se_resnet, model_name)(num_classes=num_classes)
    elif model_name[:9] == "efficient":
        return EfficientNet.from_name(model_name, num_classes=num_classes, image_size=224)
    else:
        print("Unsupported model structure.")
        exit(0)

def get_logger(args):
    log_dir = os.path.join(args.log_dir, args.expid)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_file = os.path.join(log_dir, 'log.txt')
    subscript = 1
    while os.path.exists(log_file):
        log_file = os.path.join(log_dir, 'log_{}.txt'.format(subscript))
        subscript += 1

    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
        
def get_utils(args):
    save_dir = os.path.join(args.log_dir, args.expid)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_args_path = os.path.join(save_dir, 'args.txt')
    save_dict_to_text(vars(args), save_args_path)
    residue = glob(os.path.join(save_dir, 'events*'))
    for events in residue:
        os.remove(events)
    writer = SummaryWriter(log_dir=save_dir)
    return save_dir, writer

def get_utils_cross(args):
    exp_dir = os.path.join(args.log_dir, args.expid)
    save_dir = os.path.join(exp_dir, 'fold_'+str(args.k))
    if not os.path.exists(exp_dir):
        try:
            os.mkdir(exp_dir)
        except:
            os.makedirs(exp_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_args_path = os.path.join(save_dir, 'args.txt')
    save_dict_to_text(vars(args), save_args_path)
    residue = glob(os.path.join(save_dir, 'events*'))
    for events in residue:
        os.remove(events)
    writer = SummaryWriter(log_dir=save_dir)
    return exp_dir, save_dir, writer

def get_optimizer(param, args):
    if args.op_type == 'AdamW':
        return AdamW(param, lr=args.lr, betas=(args.beta0, args.beta1), 
                     eps=args.eps, weight_decay=args.weight_decay)
    elif args.op_type == 'SGD':
        return  SGD(param, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def get_scheduler(optimizer, args):
    if args.scheduler_type == 'OneCycle':
        return OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.total_steps)
    elif args.scheduler_type == 'Cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        scheduler = MultiStepLR(optimizer, args.milestones, args.lr_decay)
    return scheduler

def get_loader(args):
    # Get dataloader for k_th fold
    dataset, loader = {}, {}
    for mode in ['train', 'val', 'test']:
        dataset[mode] = KeratitisLabeled(mode=mode, k=args.k)
        shuffle_flag = True if mode == 'train' else False
        if mode == 'train':
            if args.sampler == 'balanced':
                balanced_sampler = BalancedSampler(dataset[mode])
                loader[mode] = DataLoader(dataset[mode], sampler=balanced_sampler, batch_size=args.batch_size, 
                                        num_workers=args.num_workers, pin_memory=True)
            elif  args.sampler == 'sqrt':
                sqrt_sampler = SqrtSampler(dataset[mode])
                loader[mode] = DataLoader(dataset[mode], sampler=sqrt_sampler, batch_size=args.batch_size, 
                                        num_workers=args.num_workers, pin_memory=True)
            elif args.sampler == 'standard':
                loader[mode] = DataLoader(dataset[mode], shuffle=shuffle_flag, batch_size=args.batch_size, 
                                        num_workers=args.num_workers, pin_memory=True)
            else:
                raise Exception(f"Not supported sampler {args.sampler}")
        else:
            loader[mode] = DataLoader(dataset[mode], shuffle=shuffle_flag, batch_size=args.batch_size, 
                                    num_workers=args.num_workers, pin_memory=True)
    return dataset, loader

def load_backbone_from_args(model, args):
    if args.ckpt:
        backbone_state_dict = torch.load(args.ckpt) if os.path.exists(args.ckpt) else load_state_dict_from_url(args.ckpt)
        del backbone_state_dict['classifier.weight'], backbone_state_dict['classifier.bias'] # remove classifier param
        if args.model == 'densenet121' and not os.path.exists(args.ckpt):
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(backbone_state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    backbone_state_dict[new_key] = backbone_state_dict[key]
                    del backbone_state_dict[key]
        state_dict = model.state_dict()
        state_dict.update(backbone_state_dict)  # only update the shared backbone part from byol
        model.load_state_dict(state_dict)

def load_backbone(model, ckpt):
    backbone_state_dict = torch.load(ckpt)
    del backbone_state_dict['classifier.weight'], backbone_state_dict['classifier.bias']
    model_state_dict = model.state_dict()
    model_state_dict.update(backbone_state_dict)
    model.load_state_dict(model_state_dict)

def save_model(model, model_save_path):
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), model_save_path)
    else:
        torch.save(model.state_dict(), model_save_path)

def move_to_device(data, device):
    ret = []
    for x in data:
        if isinstance(x, torch.Tensor):
            ret.append(x.to(device))
        else:
            ret.append(x)
    return ret

def save_dict_to_text(mydict, path):
    stream = [str(k)+': '+str(v)+'\n' for k,v in mydict.items()]
    stream = reduce(lambda x, y: x+y, stream)
    with open(path, 'w') as f:
        f.writelines(stream)

def merge_report(report_list):
    N = len(report_list)  # Num of report in the list
    merged_report = {}
    for key in report_list[0].keys():
        if not isinstance(report_list[0][key], dict):
            merged_report[key] = sum([report_list[i][key] for i in range(N)]) / N
        else:
            sub_dict = {}
            for metric in report_list[0][key].keys():
                sub_dict[metric] = sum(report_list[i][key][metric] for i in range(N)) / N
            merged_report[key] = sub_dict
    return merged_report   

def convert_ckpt(state_dict):
    pattern = re.compile(r'module.(.*)')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict