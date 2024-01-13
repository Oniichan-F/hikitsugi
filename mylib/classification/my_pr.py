import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc



def plotPR(trues, probs, name_classes, force=False, save=None, prefix="", show=True, size=8):
    num_classes = len(name_classes)
    plt.figure(figsize=(size,size), facecolor='w')
    plt.title("PR-curve")
    
    # 2class
    if num_classes == 2:
        if force:
            _trues = trues.copy()
            _probs = probs.copy()
            
            _trues = [0 if x == 1 else 1 for x in _trues]
            if 0 not in trues:
                _trues.append(0)
                _probs.append(0.5)
                print("force-mode activated: add 0 behind")
            elif 1 not in trues:
                _trues.append(1)
                _probs.append(0.5)
                print("force-mode activated: add 1 behind")
            precision, recall, _ = precision_recall_curve(_trues, _probs)
            precision[-1] = 1.0
            recall[-1] = 0
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f"AUC:{pr_auc:.4f}")                
        
        else:        
            precision, recall, _ = precision_recall_curve(_trues, _probs)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f"AUC:{pr_auc:.4f}") 
    
    # multi-class
    else:
        print("unavailable")
        return
    
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim((-0.01, 1.01))
    plt.ylim((-0.01, 1.01))
    plt.legend()
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    if save is not None: plt.savefig(f"{save}/{prefix}_pr-auc.png", format="png", dpi=100)
    if show: plt.show()