import numpy as np
import pandas as pd

from PIL import Image

import torchvision.transforms.functional as TF


def loadData3(impath, lbpath, fname_key="fname", label_key="gs", resize=224, to_tensor=True):
    df = pd.read_excel(lbpath)
    images = []
    labels = df[label_key].to_list()
    fnames = df[fname_key].to_list()
    
    n = len(fnames)
    for i in range(n):
        path = f"{impath}/{fnames[i]}"
        im = Image.open(path)
        im = im.convert('RGB')
        im = im.resize((resize, resize))
        if to_tensor: im = TF.to_tensor(im)
        images.append(im)
        print(f"\rprogress {i+1}/{n}", end='')
        
    print(" >> Done")
    return images, labels, fnames


def loadData4(impath, lbpath, fname_key="fname", label_key="gs", fold_key="fold", resize=224, to_tensor=True):
    df = pd.read_excel(lbpath)
    images = []
    labels = df[label_key].to_list()
    fnames = df[fname_key].to_list()
    folds  = df[fold_key].to_list()
    
    n = len(fnames)
    for i in range(n):
        path = f"{impath}/{fnames[i]}"
        im = Image.open(path)
        im = im.convert('RGB')
        im = im.resize((resize, resize))
        if to_tensor: im = TF.to_tensor(im)
        images.append(im)
        print(f"\rprogress {i+1}/{n}", end='')
        
    print(" >> Done")
    return images, labels, fnames, folds
    

def loadData5(impath, lbpath, fname_key="fname", label1_key="gs", label2_key="gs", fold_key="fold", resize=224, to_tensor=True):
    df = pd.read_excel(lbpath)
    images  = []
    labels1 = df[label1_key].to_list()
    labels2 = df[label2_key].to_list()
    fnames  = df[fname_key].to_list()
    folds   = df[fold_key].to_list()
    
    n = len(fnames)
    for i in range(n):
        path = f"{impath}/{fnames[i]}"
        im = Image.open(path)
        im = im.convert('RGB')
        im = im.resize((resize, resize))
        if to_tensor: im = TF.to_tensor(im)
        images.append(im)
        print(f"\rprogress {i+1}/{n}", end='')
        
    print(" >> Done")
    return images, labels1, labels2, fnames, folds