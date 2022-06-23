import os, pickle
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from torchvision.transforms.transforms import RandomCrop

LABELED_DIR = "/home/ljc/Dataset/keratitis2021/"
UNLABELED_DIR= "/home/ljc/Dataset/OrigData"
#UNLABELED_DIR= "/home/ljc/Dataset/ISIC.pkl"

# This aug will apply when supvise training, after basic_aug
train_aug = transforms.Compose(
                [transforms.Resize([224, 224]),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5659, 0.3040, 0.2548),(0.0773, 0.0613, 0.0561))]
            )

# This aug will apply when supvise validation and testing, after basic_aug
test_aug =  transforms.Compose(
                [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize((0.5659, 0.3040, 0.2548),(0.0773, 0.0613, 0.0561))]
            )

# Aug for unsupervised only, mean&std calculated on unlabeled keratiits dataset
unsup_aug = transforms.Compose(
                [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize((0.5444, 0.3120, 0.2602),(0.2240, 0.1733, 0.1637))]
            )

# Aug for semi-supervised learning
semisl_aug = transforms.Compose(
                [transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop([224, 224]),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.5659, 0.3040, 0.2548),(0.0773, 0.0613, 0.0561))]
            )

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def merge_split(index2dict: dict, indexes: list) -> dict:
    merged_dict = index2dict[str(indexes[0])]
    for index in indexes[1:]:
        for key in merged_dict.keys():
            merged_dict[key].extend(index2dict[str(index)][key])
    return merged_dict

class KeratitisLabeled(Dataset):
    def __init__(self, root=LABELED_DIR, mode='train', k=0, transform=None, target_transform=None, loader=default_loader):
        data_info_path = os.path.join(root, 'datainfo.pkl')
        with open(data_info_path, 'rb') as f:
            data_info = pickle.load(f)[k]  # Load k_th data division
        #print(data_info.keys())  
        dis2pids = data_info['{}_pid_dict'.format(mode)]
        dis2imgs = data_info['{}_img_dict'.format(mode)]
  
        classes = sorted(dis2pids.keys())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        if transform is None:
            transform = train_aug if mode == 'train' else test_aug

        self.root = root
        self.loader = loader
        self.mode = mode
        self.k = k
        self.transform = transform
        self.target_transform = target_transform
        self.dis2pids = dis2pids
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.dis2imgs = dis2imgs
        self.samples = self.makedataset()

    def makedataset(self):
        # Extend img_path to absolute full_path
        instances = []
        for dis, imgs in self.dis2imgs.items():
            for img in imgs:
                img_fullpath = os.path.join(self.root, 'images', img)
                instances.append((img_fullpath, dis))
        return sorted(instances)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        target = self.class_to_idx[target]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
 
class KeratitisUnlabeled(Dataset):
    def __init__(self, root=UNLABELED_DIR, loader=default_loader, transform=None):
        annotation_path = os.path.join(root, "selected.pkl")
        with open(annotation_path, 'rb') as f:
            annotation = pickle.load(f)
        samples = annotation['selected']

        self.root = root
        self.loader = loader
        self.samples = samples
        self.transform = transform if transform else unsup_aug

    def __getitem__(self, index):
        path = self.samples[index]
        full_path = os.path.join(self.root, path)
        sample = self.loader(full_path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, index

    def __len__(self):
        return len(self.samples)

class ISICUnlabeled(Dataset):
    def __init__(self, root=UNLABELED_DIR, loader=default_loader, transform=None):
        with open(root, 'rb') as f:
            samples = pickle.load(f)

        self.root = root
        self.loader = loader
        self.samples = samples
        self.transform = transform if transform else unsup_aug

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, index

    def __len__(self):
        return len(self.samples)

class BalancedSampler(Sampler):
    def __init__(self, dataset, class_len=None):
        self.dataset = dataset
        if class_len:
            self.num_classes = len(class_len)
            self.length = class_len
        else:
            self.num_classes =  len(dataset.classes)
            self.length = [len(instances) for instances in sorted(dataset.dis2imgs.values())]
        self.seperator = [sum(self.length[:k]) for k in range(len(self.length) + 1)]

    def __iter__(self):
        for _ in range(len(self.dataset)):
            cidx = torch.randint(0, self.num_classes, (1,)).item()
            yield torch.randint(self.seperator[cidx], self.seperator[cidx + 1], (1,)).item()

    def __len__(self):
        return len(self.dataset)    

class SqrtSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_classes = len(dataset.classes)
        self.length = [len(instances) for instances in dataset.dis2imgs.values()]
        self.seperator = [sum(self.length[:k]) for k in range(len(self.length) + 1)]

    def __iter__(self):
        for _ in range(len(self.dataset)):
            weight = torch.sqrt(torch.FloatTensor(self.length))
            cidx = torch.multinomial(weight, 1, True).item()
            yield torch.randint(self.seperator[cidx], self.seperator[cidx + 1], (1,)).item()

    def __len__(self):
        return len(self.dataset) 

def main():
    k = 0
    trainset = KeratitisLabeled(mode='train', k=k)
    testset = KeratitisLabeled(mode='test', k=k)
    print(f'trainset length: {len(trainset)}, testset length: {len(testset)}')
    """
    for dataset in [trainset, testset]:
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
        img, target =  next(iter(dataloader))
        print(img.shape, target.shape)
    """
    
    sampler = BalancedSampler(trainset)
    dataloader = DataLoader(trainset, sampler=sampler, batch_size=128, num_workers=8)
    from collections import Counter
    cnt = Counter()
    for img, target in dataloader:
        cnt.update(target.tolist())
        print(cnt)
        cnt.clear()
    

if __name__ == '__main__':
    main()