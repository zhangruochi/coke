import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import patches 

font = {'family': 'serif',
        # 'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }


def sample_images(x,y, shape):
    """
    input: 
        x(Tensor[num_images, rows, columns]): images tensor
        y(array): labels
        shape(tuple): (rows,col) 
    output:
        grid of smaple images
    """
    sample_index = random.sample(range(x.shape[0]),shape[0]*shape[1])
    
    fig, axs = plt.subplots(nrows=shape[0], ncols=shape[1], figsize = (12,8))

    samples = [[x[idx],y[idx]] for idx in sample_index]
    index = 0

    for row in axs:
        for ax in row:
            ax.imshow(samples[index][0])
            ax.set_xlabel(samples[index][1], fontdict = font)
            index+=1

    plt.subplots_adjust(wspace = 0.2, hspace = 0.5) 
    plt.show()



def show_batch(x,y,shape = None):
    """
    input: 
        x(Tensor[num_images, rows, columns]): images tensor
        y(array): labels
        shape(tuple): (rows,col) 
    output:
        grid of smaple images
    """

    if not shape:
        shape = (int(x.shape[0]**0.5), int(x.shape[0]**0.5))

    fig, axs = plt.subplots(nrows= shape[0], ncols=shape[1], figsize = (12,8))
    index = 0
    for row in axs:
        for ax in row:
            ax.imshow(x[index])
            ax.set_xlabel(y[index], )
            index+=1

    # plt.subplots_adjust(wspace = 0.2, hspace = 0.5) 
    fig.tight_layout()
    plt.show()




def show_object_box(img, mask):
    obj_ids = np.unique(mask)[1:]
    num_objs = len(obj_ids)
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    for i in range(num_objs):
        pos = np.where(mask == obj_ids[i])
        x_min = np.min(pos[1])
        x_max = np.max(pos[1])
        y_min = np.min(pos[0])
        y_max = np.max(pos[0])
        rect = patches.Rectangle((x_min,y_min),x_max - x_min,y_max - y_min,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()


