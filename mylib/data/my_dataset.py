import numpy as np

from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import Subset


class MyDataset(Dataset):
    def __init__(self, images, labels, fnames, name_classes):
        self.images       = images
        self.labels       = labels
        self.fnames       = fnames
        self.name_classes = name_classes
        self.num_classes  = len(name_classes)
        self.count()
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.fnames[idx]
    
    def count(self):
        counter = np.zeros(self.num_classes)
        for label in self.labels:
            counter[label] += 1
            
        print("--- Dataset element counter ---")
        for i in range(self.num_classes):
            print(f"{self.name_classes[i]} : {int(counter[i])}")
            
    def getRatio(self):
        counter = np.zeros(self.num_classes)
        for label in self.labels:
            counter[label] += 1
        counter /= len(self.labels)

        return counter    
    

def getKFoldManager(x, y, n):
    manager = []
    skf = StratifiedKFold(n_splits=n, shuffle=True)
    
    fold_id   = 0
    fold_name = 1
    for train_idx, test_idx in skf.split(x, y):
        d = dict(
            fold_id   = fold_id,
            fold_name = fold_name,
            train     = train_idx,
            test      = test_idx
        )
        
        print(f"fold:{d['fold_name']} train:{len(d['train'])} / test:{len(d['test'])}")
        manager.append(d)
        fold_id   += 1
        fold_name += 1
    
    return manager


def getLocoManager(folds):
    managers   = []
    fold_names = list(set(folds))
    
    fold_id = 0
    for fold_name in fold_names:
        train_idx, test_idx = [], []
        for idx in range(len(folds)):
            if folds[idx] == fold_name:
                test_idx.append(idx)
            else:
                train_idx.append(idx)
                
        d = dict(
            fold_id   = fold_id,
            fold_name = fold_name,
            train     = train_idx,
            test      = test_idx
        )
        
        print(f"fold:{d['fold_name']} train:{len(d['train'])} / test:{len(d['test'])}")
        managers.append(d)
        fold_id += 1
        
    return managers


def getLoader(dataset, manager, batch_size, val_ratio=0.1):
    train_idx = manager['train']
    test_idx  = manager['test']
    
    trainval_ds  = Subset(dataset, train_idx)
    trainval_num = len(trainval_ds)
    valid_num    = int(trainval_num * val_ratio)
    train_num    = trainval_num - valid_num
    train_ds, valid_ds = random_split(trainval_ds, [train_num, valid_num], generator=torch.Generator().manual_seed(42))
    test_ds = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    loader = dict(
        fold  = manager['fold_name'],
        train = train_loader,
        valid = valid_loader,
        test  = test_loader
    )

    print(f"train:{len(train_ds)} / valid:{len(valid_ds)} / test:{len(test_ds)}")
    return loader


def getLabelRatioInTrain(dataset, manager, num_classes):
    train_idx = manager['train']
    train_ds = Subset(dataset, train_idx)

    counter = np.zeros(num_classes)
    for i in range(len(train_ds)):
        _, y, _ = train_ds[i]
        counter[y] += 1
    counter /= len(train_ds)
    return counter
            
    