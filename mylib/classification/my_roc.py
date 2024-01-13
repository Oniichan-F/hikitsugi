import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc, roc_auc_score


def plotROC(trues, probs, name_classes, force=False, save=None, prefix="", size=8):
    num_classes = len(name_classes)
    plt.figure(figsize=(size,size), facecolor='w')
    plt.title("ROC-curve")
    
    # 2class
    if num_classes == 2:
        if force:
            _trues = trues.copy()
            _probs = probs.copy()
            
            if 0 not in trues:
                _trues.append(0)
                _probs.append(0.5)
                print("force-mode activated: add 0 behind")
            elif 1 not in trues:
                _trues.append(1)
                _probs.append(0.5)
                print("force-mode activated: add 1 behind")
            fpr, tpr, _ = roc_curve(_trues, _probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"AUC:{roc_auc:.4f}")                
        
        else:        
            fpr, tpr, _ = roc_curve(trues, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"AUC:{roc_auc:.4f}")
    
    # multi-class
    else:
        trues_onehot = F.one_hot(torch.tensor(trues), num_classes=num_classes).numpy()
        fprs, tprs, roc_aucs = {}, {}, {}
        base_fpr = np.linspace(0, 1, 1024)
        for i in range(num_classes):
            fprs[i], tprs[i], _ = roc_curve(trues_onehot[:,i], np.array(probs)[:,i])
            roc_aucs[i] = auc(fprs[i], tprs[i])
        mean_tpr = np.zeros_like(base_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(base_fpr, fprs[i], tprs[i])
        mean_tpr /= num_classes
        mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
        roc_auc = roc_auc_score(trues_onehot, probs, average='macro', multi_class='ovr')
        for i in range(num_classes):
            line = f"{name_classes[i]} (AUC:{roc_aucs[i]:.4f})"
            plt.plot(fprs[i], tprs[i], label=line, lw=1.5, alpha=0.75)
        plt.plot(base_fpr, mean_tpr, label=f"macro-mean (AUC:{roc_auc:.4f})", lw=3, color='black')
    
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend()
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    if save is not None: plt.savefig(f"{save}/{prefix}_roc-auc.png", format="png", dpi=100)
    plt.show()
    
def plotROCwithYoudenPoint(trues, probs, name_classes, y_fpr, y_tpr, force=False, save=None, prefix="", show=True, size=8):
    num_classes = len(name_classes)
    plt.figure(figsize=(size,size), facecolor='w')
    plt.title("ROC-curve")
    
    # 2class
    if num_classes == 2:
        if force:
            _trues = trues.copy()
            _probs = probs.copy()
            
            if 0 not in trues:
                _trues.append(0)
                _probs.append(0.5)
                print("force-mode activated: add 0 behind")
            elif 1 not in trues:
                _trues.append(1)
                _probs.append(0.5)
                print("force-mode activated: add 1 behind")
            fpr, tpr, _ = roc_curve(_trues, _probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"AUC:{roc_auc:.4f}")                
        
        else:        
            fpr, tpr, _ = roc_curve(trues, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"AUC:{roc_auc:.4f}")
    
    plt.plot(y_fpr, y_tpr, marker='*', markersize=15, color='r')
    
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend()
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    if save is not None: plt.savefig(f"{save}/{prefix}_roc-auc.png", format="png", dpi=100)
    if show: plt.show()
    
    
def plotDistributionHistogram(trues, probs, save=None, prefix="", show=True):
    df = pd.DataFrame({
        'true' : trues,
        'prob' : probs,
    })
    
    x0 = df[df['true']==0]['prob']
    x1 = df[df['true']==1]['prob']
    
    plt.figure(figsize=(12,6), facecolor='w')
    plt.title("Probability Distribution")
    plt.hist([x0, x1], color=['red', 'blue'], bins=50, stacked=True)
    plt.xticks(np.arange(0, 1, 0.05))
    plt.xlim(min(probs), max(probs))
    plt.xlabel("probability (negative <--  --> positive)")
    plt.ylabel("frequancy")
    plt.grid()
    if save is not None: plt.savefig(f"{save}/{prefix}_distribution.png", format="png", dpi=100)
    if show: plt.show()
    