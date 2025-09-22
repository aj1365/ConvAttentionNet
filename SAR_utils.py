# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:21:37 2022

@author: malkhatib
"""
import scipy.io as sio
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
from operator import truediv
import random 
from sklearn.utils import shuffle




def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]),dtype=('complex64'))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=('complex64'))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def target(name):
    if name == 'FL' or name == 'FL':
        target_names = ['Unassigned', 'Water', 'Forest', 'Lucerne', 'Grass', 'Rapeseed',
                        'Beet', 'Potatoes', 'Peas', 'Stem Beans', 'Bare Soil', 'Wheat', 'Wheat 2', 
                        'Wheat 3', 'Barley', 'Buildings']
    elif name == 'SF':
        target_names = ['Unassigned', 'Bare Soil', 'Mountain', 'Water', 'Urban', 'Vegetation']
        
    return target_names 
    
def num_classes(dataset):
    if dataset == 'FL':
        output_units = 15
    elif dataset == 'SF':
        output_units = 5
    elif dataset == 'ober' or dataset == 'ober_real':
        output_units = 3
    
    return output_units




def Patch(data,height_index,width_index, PATCH_SIZE):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    
    return patch

def getTrainTestSplit(X_cmplx, y, pxls_num):
    if type(pxls_num) != list:
        pxls_num = [pxls_num]*len(np.unique(y))
        
    if len(np.unique(y)) != len(pxls_num):
        print("length of pixels list doen't match the number of classes in the dataset")
        return
    else:
        xTrain_cmplx = []
        yTrain = []
        
        xTest_cmplx  = []
        yTest  = []
        for i in range(len(np.unique(y))):
            if pxls_num[i] > len(y[y==i]):
                print("Number of training pixles is larger than total class pixels")
                return
            else:
                random.seed(321) #optional to reproduce the data
                samples = random.sample(range(len(y[y==i])), pxls_num[i])
                xTrain_cmplx.extend(X_cmplx[y==i][samples])
                
                yTrain.extend(y[y==i][samples])
                
                tmp1 = list(X_cmplx[y==i])
               
                tmp3 = list(y[y==i])
                for ele in sorted(samples, reverse = True):
                    del tmp1[ele]
                    del tmp3[ele]

                xTest_cmplx.extend(tmp1)
                yTest.extend(tmp3)
     
  
    xTrain_cmplx, yTrain = shuffle(xTrain_cmplx, yTrain, random_state=321)  
    xTest_cmplx,  yTest = shuffle(xTest_cmplx, yTest, random_state=345)
    
    #xTrain_rgb, yTrain = shuffle(xTrain_rgb, yTrain, random_state=321)  
    #xTest_rgb, yTest = shuffle(xTest_rgb, yTest, random_state=345)
    
    
    
    xTrain_cmplx = np.array(xTrain_cmplx)
    yTrain = np.array(yTrain)
    
    xTest_cmplx = np.array(xTest_cmplx)
    yTest = np.array(yTest)
    
      
    return xTrain_cmplx, yTrain, xTest_cmplx, yTest
        
        
    
import cvnn.layers as complex_layers
def cmplx_SE_Block(xin, se_ratio = 8):
    # Squeeze Path
    xin_gap =  GlobalCmplxAveragePooling2D(xin)
    sqz = complex_layers.ComplexDense(xin.shape[-1]//se_ratio, activation='cart_relu')(xin_gap)
    
    # Excitation Path
    excite1 = complex_layers.ComplexDense(xin.shape[-1], activation='cart_sigmoid')(sqz)
    
    out = tf.keras.layers.multiply([xin, excite1])
    
    return out
    
   

import tensorflow as tf
def GlobalCmplxAveragePooling2D(inputs):
    inputs_r = tf.math.real(inputs)
    inputs_i = tf.math.imag(inputs)
    
    output_r = tf.keras.layers.GlobalAveragePooling2D()(inputs_r)
    output_i = tf.keras.layers.GlobalAveragePooling2D()(inputs_i)
    
    if inputs.dtype == 'complex' or inputs.dtype == 'complex64':
           output = tf.complex(output_r, output_i)
    else:
           output = output_r
    
    return output




def Standardize_data(X):
    new_X = np.zeros(X.shape, dtype=(X.dtype))
    _,_,c = X.shape
    for i in range(c):
        new_X[:,:,i] = (X[:,:,i] - np.mean(X[:,:,i])) / np.std(X[:,:,i])
        
    return new_X
        
        



from numpy.fft import fft2, fftshift
def getFFT(X):
    X_fft = np.zeros(X.shape, dtype='complex64')
    for ii in range(len(X)):
        for jj in range(X.shape[3]):
            X_fft[ii,:,:,jj] = fftshift(fft2(X[ii,:,:,jj])) 
            #X_fft[ii,:,:,jj] = fftshift(fft2(X[ii,:,:,jj])) 
            
            
    return X_fft


import keras
def cart_gelu(x):
    x_r = tf.math.real(x)
    x_i = tf.math.imag(x)
    
    gelu_r = keras.activations.gelu(x_r, approximate=False)
    gelu_i = keras.activations.gelu(x_i, approximate=False)
    
    if x.dtype == 'complex' or x.dtype == 'complex64':
           output = tf.complex(gelu_r, gelu_i)
    else:
           output = gelu_r
    
    return output


def predict_by_batching(model, input_tensor, batch_size):
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
    
    num_samples = input_tensor.shape[0]
    k = 0
    predictions = []
    for i in range(0, num_samples, batch_size):
        print("batch", k, " out of", num_samples//batch_size)
        print(k*batch_size, "out of", num_samples )
        k+=1
        batch = input_tensor[i:i + batch_size]
        batch_predictions = model.predict(batch, verbose=1)
        predictions.append(batch_predictions)
        
    Y_pred_test = np.concatenate(predictions, axis=0)
  
    return Y_pred_test
















