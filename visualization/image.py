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

def grad_cam(input_model, image, category_index, layer_name):
    """
    GradCAM method for visualizing input saliency.
    
    Args:
        input_model (Keras.model): model to compute cam for
        image (tensor): input to model, shape (1, H, W, 3)
        cls (int): class to compute cam with respect to
        layer_name (str): relevant layer in model
        H (int): input height
        W (int): input width
    Return:
        cam ()
    """
    cam = None
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # 1. Get placeholders for class output and last layer
    # Get the model's output
    output_with_batch_dim = model.output
    
    # Remove the batch dimension
    output_all_categories = output_with_batch_dim[0]
    
    # Retrieve only the disease category at the given category index
    y_c = output_all_categories[category_index]
    
    # Get the input model's layer specified by layer_name, and retrive the layer's output tensor
    spatial_map_layer = model.get_layer(layer_name).output

    # 2. Get gradients of last layer with respect to output

    # get the gradients of y_c with respect to the spatial map layer (it's a list of length 1)
    grads_l = K.gradients(y_c,spatial_map_layer)
    
    # Get the gradient at index 0 of the list
    grads = grads_l[0]
        
    # 3. Get hook for the selected layer and its gradient, based on given model's input
    # Hint: Use the variables produced by the previous two lines of code
    spatial_map_and_gradient_function = K.function([input_model.input], [spatial_map_layer, grads])
    
    # Put in the image to calculate the values of the spatial_maps (selected layer) and values of the gradients
    spatial_map_all_dims, grads_val_all_dims = spatial_map_and_gradient_function([image])

    # Reshape activations and gradient to remove the batch dimension
    # Shape goes from (B, H, W, C) to (H, W, C)
    # B: Batch. H: Height. W: Width. C: Channel    
    # Reshape spatial map output to remove the batch dimension
    spatial_map_val = spatial_map_all_dims[0]
    
    # Reshape gradients to remove the batch dimension
    grads_val = grads_val_all_dims[0]
    
    # 4. Compute weights using global average pooling on gradient 
    # grads_val has shape (Height, Width, Channels) (H,W,C)
    # Take the mean across the height and also width, for each channel
    # Make sure weights have shape (C)
    weights = grads_val.mean(axis=(0,1))
    # 5. Compute dot product of spatial map values with the weights
    cam = np.dot(spatial_map_val,weights)

    ### END CODE HERE ###
    
    # We'll take care of the postprocessing.
    H, W = image.shape[1], image.shape[2]
    cam = np.maximum(cam, 0) # ReLU so we only get positive importance
    cam = cv2.resize(cam, (W, H), cv2.INTER_NEAREST)
    cam = cam / cam.max()

    return cam


def compute_gradcam(model, img, mean, std, data_dir, df, 
                    labels, selected_labels, layer_name='conv5_block16_concat'):
    """
    Compute GradCAM for many specified labels for an image. 
    This method will use the `grad_cam` function.
    
    Args:
        model (Keras.model): Model to compute GradCAM for
        img (string): Image name we want to compute GradCAM for.
        mean (float): Mean to normalize to image.
        std (float): Standard deviation to normalize the image.
        data_dir (str): Path of the directory to load the images from.
        df(pd.Dataframe): Dataframe with the image features.
        labels ([str]): All output labels for the model.
        selected_labels ([str]): All output labels we want to compute the GradCAM for.
        layer_name: Intermediate layer from the model we want to compute the GradCAM for.
    """
    img_path = data_dir + img
    preprocessed_input = load_image_normalize(img_path, mean, std)
    predictions = model.predict(preprocessed_input)
    print("Ground Truth: ", ", ".join(np.take(labels, np.nonzero(df[df["Image"] == img][labels].values[0]))[0]))

    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(load_image(img_path, df, preprocess=False), cmap='gray')
    
    j = 1
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###    
    # Loop through all labels
    for i in range(len(labels)): # complete this line
        # Compute CAM and show plots for each selected label.
        
        # Check if the label is one of the selected labels
        if labels[i] in selected_labels: # complete this line
            
            # Use the grad_cam function to calculate gradcam
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            
            ### END CODE HERE ###
            
            print("Generating gradcam for class %s (p=%2.2f)" % (labels[i], round(predictions[0][i], 3)))
            plt.subplot(151 + j)
            plt.title(labels[i] + ": " + str(round(predictions[0][i], 3)))
            plt.axis('off')
            plt.imshow(load_image(img_path, df, preprocess=False), cmap='gray')
            plt.imshow(gradcam, cmap='magma', alpha=min(0.5, predictions[0][i]))
            j +=1
