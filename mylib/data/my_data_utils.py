import random
import matplotlib.pyplot as plt


def plotImage(im, title="", fontsize=12, is_tensor=True):
    if is_tensor: im = im.permute(1, 2, 0)
    
    plt.figure(facecolor='w')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.xlabel(title, fontsize=fontsize)
    plt.imshow(im, cmap=plt.cm.binary)
    
    
def plotImageSamples(ims, is_tensor=True):
    SAMPLE_NUM = 8
    
    def plot_image(im, is_tensor):
        if is_tensor: im = im.permute(1, 2, 0)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(im, cmap=plt.cm.binary)
    
    samples = random.sample(ims, SAMPLE_NUM)
    plt.figure(figsize=(2*SAMPLE_NUM, 8*SAMPLE_NUM), facecolor='w')
    for i in range(SAMPLE_NUM):
        plt.subplot(1, SAMPLE_NUM, i+1)
        plot_image(ims[i], is_tensor=is_tensor)
    
    plt.show()
        
    