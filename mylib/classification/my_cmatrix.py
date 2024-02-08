import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve


def plotConfusionMatrix(trues, probs, name_classes, t=0.5, save_string=None, show=True):
    num_classes = len(name_classes)
    
    if num_classes == 2:
        preds = [0 if t <= x else 1 for x in probs]
        cmatrix = get_cmatrix(trues, preds, num_classes)
    else:
        print("not avaiable")
    
    fig, ax = plt.subplots()
    im = ax.imshow(cmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks      = np.arange(cmatrix.shape[1]),
        yticks      = np.arange(cmatrix.shape[0]),
        xticklabels = name_classes,
        yticklabels = name_classes,
        title       = "Confusion Matrix",
        xlabel      = "pred label",
        ylabel      = "true label",
    )
    
    thresh = cmatrix.max() / 2
    for i in range(cmatrix.shape[0]):
        for j in range(cmatrix.shape[1]):
            ax.text(j, i, int(cmatrix[i,j]),
                    ha='center', va='center',
                    color = 'white' if cmatrix[i,j]>thresh else 'black' 
            )
    
    fig.tight_layout()
    plt.xlim(-0.5, num_classes-0.5)
    plt.ylim(num_classes-0.5, -0.5)
    if save_string is not None: plt.savefig(save_string, format="png", dpi=100)
    if show: plt.show()


def get_youden_index(trues, probs):
    fpr, tpr, thresh = roc_curve(trues, probs)
    index = np.argmax(tpr - fpr)
    youden_thresh = 1 -thresh[index]
    print(f"Youden-index: {youden_thresh}") 
    return youden_thresh, fpr[index], tpr[index]


def get_cmatrix(trues, preds, num_classes):
    cmatrix = np.zeros((num_classes, num_classes))
    for i in range(len(trues)):
        cmatrix[trues[i]][preds[i]] += 1     
    return cmatrix