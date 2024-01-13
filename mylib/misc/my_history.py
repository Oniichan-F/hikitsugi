import pandas as pd
import matplotlib.pyplot as plt


def plotHistory(d, save=None, size=(18,6)):
    train_loss = d["train_loss"]
    valid_loss = d["valid_loss"]
    train_acc  = d["train_acc"]
    valid_acc  = d["valid_acc"]
    
    num_epochs = range(len(train_loss))
    plt.figure(figsize=size, facecolor='w')
    
    plt.subplot(1, 2, 1)
    plt.plot(num_epochs, train_loss, label="train", color='blue')
    plt.plot(num_epochs, valid_loss, label="valid", color='orange')
    plt.title("Loss")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(num_epochs, train_acc, label="train", color='blue')
    plt.plot(num_epochs, valid_acc, label="valid", color='orange')
    plt.title("Accuracy")
    plt.grid()
    plt.legend()
    
    if save is not None: plt.savefig(save, format="png", dpi=100)
    plt.show()