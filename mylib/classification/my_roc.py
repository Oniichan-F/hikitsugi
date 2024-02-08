import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc, roc_auc_score

    
def plotROCwithYoudenPoint(trues, probs, name_classes, youden_fpr, youden_tpr, save_string=None, show=True, size=8):
    num_classes = len(name_classes)
    if num_classes != 2:
        print("2-class only")
        return
    if 0 not in trues or 1 not in trues:
        print("cannot draw ROC-curve")
        return
    
    plt.figure(figsize=(size,size), facecolor='w')
    plt.title("ROC-curve")
        
    fpr, tpr, _ = roc_curve(trues, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC:{roc_auc:.4f}")                
    plt.plot(youden_fpr, youden_tpr, marker='*', markersize=15, color='r')
    
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend()
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    if save_string is not None: plt.savefig(save_string, format="png", dpi=100)
    if show: plt.show()
    
    
def plotDistributionHistogram(trues, probs, save_string=None, show=True):
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
    if save_string is not None: plt.savefig(save_string, format="png", dpi=100)
    if show: plt.show()
    