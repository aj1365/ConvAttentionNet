# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:21:37 2022

@author: malkhatib
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import spectral
import matplotlib.pyplot as plt
import scipy.io as sio
from operator import truediv

def get_img_indexes (class_map, removeZeroindexes = True):
    """
    Get indices of elements in the class map.
    
    Parameters:
    class_map (numpy array): The class map (2D array).
    removeZero (bool): If True, return indices of non-zero elements, 
                       otherwise return indices of all elements.
    
    Returns:
    tuple: (indices, labels)
           - indices: List of tuples representing the indices of the selected elements.
           - labels: Array of labels corresponding to the indices.
    """
    if removeZeroindexes:
        # Get indices of non-zero values
        indices = np.argwhere(class_map != 0)
    else:
        # Get indices of all elements (including zeros)
        indices = np.argwhere(class_map != None)
    
    # Flatten the class map to get the corresponding pixel values (labels)
    labels = class_map[indices[:, 0], indices[:, 1]]
    
    # Convert indices to a list of tuples for easier use
    indices = [tuple(idx) for idx in indices]
    
    return indices, np.array(labels.tolist()) - 1


def createImageCubes(X, indices, windowSize):
    """
    Extract patches centered at given indices from the hyperspectral image 
    after applying zero padding.
    
    Parameters:
    X (numpy array): Hyperspectral image of shape (N, M, P)
    indices (list of tuples): List of indices where patches should be extracted
    windowSize (int): Window size, the patch will be of size (windowSize, windowSize)
    
    Returns:
    list: List of image patches extracted from the padded hyperspectral image
    """
    # Calculate margin based on window size
    margin = windowSize // 2
    
    # Apply zero padding to the hyperspectral image
    N, M, P = X.shape
    X_padded = np.zeros((N + 2 * margin, M + 2 * margin, P))
    
    # Offsets to place the original image in the center of the padded image
    x_offset = margin
    y_offset = margin
    X_padded[x_offset:N + x_offset, y_offset:M + y_offset, :] = X
    
    # Extract patches centered at the provided indices
    patches = []
    
    for idx in indices:
        i, j = idx
        i = i + margin
        j = j + margin
        # Get patch boundaries, ensuring the patch is centered at (i, j)
        i_min = i - margin  # Centered on the index, accounting for padding
        i_max = i_min + windowSize
        j_min = j - margin
        j_max = j_min + windowSize
        
        # Extract the patch
        patch = X_padded[i_min:i_max, j_min:j_max, :]
        

        patches.append(patch)
    
    return np.array(patches)


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



def num_classes(dataset):
    if dataset == 'FL':
        output_units = 15
    elif dataset == 'SF':
        output_units = 5
    elif dataset == 'ober' or dataset == 'ober_real':
        output_units = 3
    return output_units

def loadData(name):
    data_path = 'Datasets/PolSARData/'
    if name == 'FL':
        data = sio.loadmat(os.path.join(data_path, 'Flevoland_T3RF.mat'))['T3RF']
        labels = sio.loadmat(os.path.join(data_path, 'FlevoLand_gt.mat'))['gt']
        class_labels = ['Water', 'Forest', 'Lucerne', 'Grass', 'Rapeseed',
                        'Beet', 'Potatoes', 'Peas', 'Stem Beans', 'Bare Soil', 'Wheat', 'Wheat 2', 
                        'Wheat 3', 'Barley', 'Buildings']

    if name == 'SF':
        data = sio.loadmat(os.path.join(data_path, 'SanFrancisco_T3RF.mat'))['T3RF']
        labels = sio.loadmat(os.path.join(data_path, 'SanFrancisco_gt.mat'))['SanFrancisco_gt']
        class_labels = ['Bare Soil', 'Mountain', 'Water', 'Urban', 'Vegetation']
    
    if name == 'ober':
        data = sio.loadmat(os.path.join(data_path, 'Oberpfaffenhofen_T3RF.mat'))['T3RF']
        labels = sio.loadmat(os.path.join(data_path, 'Oberpfaffenhofen_gt.mat'))['gt']
        class_labels = ["Build-Up Areas", "Wood Land", "Open Areas"]
        
    return data, labels, class_labels



def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def predict_by_batching(model, input_tensor_idx, batch_size, X, windowSize):
    '''
    Function to to perform predictions by dividing large tensor into small ones 
    to reduce load on GPU
    
    Parameters
    ----------
    model: The model itself with pre-trained weights.
    input_tensor: Tensor of diemnsion batches x windowSize x windowSize x channels x 1.
    batch_size: integer value smaller than batches .

    Returns
    -------
    Predicetd labels
    '''
    
    num_samples = len(input_tensor_idx)
    k = 0
    predictions = []
    for i in tqdm(range(0, num_samples, batch_size), desc="Progress"):
        k+=1
        
        batch = createImageCubes(X, input_tensor_idx[i:i + batch_size], windowSize)
        batch_predictions = model.predict(batch, verbose=0)
        predictions.append(batch_predictions)
        
    Y_pred_test = np.concatenate(predictions, axis=0)
  
    return Y_pred_test

def get_class_map(model, X, label, window_size):
    indexes, labels = get_img_indexes(label, removeZeroindexes = False)
    
    y_pred = predict_by_batching(model, indexes, 10000, X, window_size)
    
    y_pred = (np.argmax(y_pred, axis=1)).astype(np.uint8)
    
    Y_pred = np.reshape(y_pred, label.shape) + 1


    gt_binary = label

    gt_binary[gt_binary>0]=1
    
   
    return Y_pred




def img_display(data = None, rgb_band = None, classes = None,class_name = None,title = None, 
                figsize = (7,7),palette = spectral.spy_colors):
    if data is not None:
        im_rgb = np.zeros_like(data[:,:,0:3])
        im_rgb = data[:,:,rgb_band]
        im_rgb = im_rgb/(np.max(np.max(im_rgb,axis = 1),axis = 0))*255
        im_rgb = np.asarray(im_rgb,np.uint8)
        fig, rgbax = plt.subplots(figsize = figsize)
        rgbax.imshow(im_rgb)
        rgbax.set_title(title)
        rgbax.axis('off')
        
    elif classes is not None:
        rgb_class = np.zeros((classes.shape[0],classes.shape[1],3))
        for i in np.unique(classes):
            rgb_class[classes==i]=palette[i]
        rgb_class = np.asarray(rgb_class, np.uint8)
        _,classax = plt.subplots(figsize = figsize)
        classax.imshow(rgb_class)
        classax.set_title(title)
        classax.axis('off')
        




