import keras
from keras.layers import Conv2D, Dense, Reshape, BatchNormalization
from keras.layers import Dropout, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral
from keras import layers
import tensorflow as tf
# %matplotlib inline
import tensorflow_addons as tfa


from SAR_utils import num_classes, predict_by_batching


def loadData(name):
    data_path = 'Datasets/PolSARData/'
    if name == 'FL':

        data = sio.loadmat(os.path.join(data_path, 'Flevoland_T3RF.mat'))['T3RF']
        labels = sio.loadmat(os.path.join(data_path, 'FlevoLand_gt.mat'))['gt']

    if name == 'SF':

        data = sio.loadmat(os.path.join(data_path, 'SanFrancisco_T3RF.mat'))['T3RF']
        labels = sio.loadmat(os.path.join(data_path, 'SanFrancisco_gt.mat'))['SanFrancisco_gt']
    
    if name == 'ober':
        data = sio.loadmat(os.path.join(data_path, 'Oberpfaffenhofen_T3RF.mat'))['T3RF']
        labels = sio.loadmat(os.path.join(data_path, 'Oberpfaffenhofen_gt.mat'))['gt']
        
    return data, labels

def linear_attention_map(inputs, out_channel):
    # Global average pooling to reduce spatial dimensions
    avg_pool = layers.GlobalAveragePooling2D(keepdims=True)(inputs)

    # First convolution with linear kernel size (6, 1) for horizontal filtering
    conv2 = layers.Conv2D(out_channel // 2, kernel_size=(6, 1), padding='same', use_bias=True)(avg_pool)
    act2 = layers.Activation("relu")(conv2)

    # Second convolution with linear kernel size (1, 6) for vertical filtering
    conv3 = layers.Conv2D(out_channel // 2, kernel_size=(1, 6), padding='same', use_bias=True)(act2)
    act3 = layers.Activation("relu")(conv3)

    # Final 1x1 convolution to restore the output channels
    conv4 = layers.Conv2D(out_channel, kernel_size=1, use_bias=True)(act3)
    attention_map = layers.Activation("sigmoid")(conv4)

    return attention_map

    
def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def createImageCubes(X, y, windowSize=8, removeZeroLabels = True):
    margin = int((windowSize) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin , c - margin:c + margin ]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


## GLOBAL VARIABLES
dataset = 'FL'
test_ratio = 0.99
windowSize = 12

    


X , Y = loadData(dataset)


X = (X-np.min(X))/(np.max(X)-np.min(X))

X1, Y1 = createImageCubes(X, Y, windowSize=windowSize)

X_train, X_test, y_train, y_test = splitTrainTestSet(X1, Y1, test_ratio)


total = 0
numm = []
for i in range(int(np.max(y_test)+1)):
    tmp = np.sum(y_train==i)
    total = total + tmp
    numm.append(tmp)
    print("Class #"+str(i) +": " + str(tmp))
    
    
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


from tensorflow.keras import layers
image_size=12

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    #name="data_augmentation",
)

# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(X_train)

def mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    residuals = x
    
    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x)
    pos_emb3 = layers.Conv2D(filters, kernel_size=1)(x)
    x = keras.layers.Add()([residuals, pos_emb1, pos_emb2, pos_emb3])

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])

    return x

def ConvAttentionNet(image_size=windowSize, filters=128, 
                     depth=4, kernel_size=3, 
                     patch_size=2, num_classes=num_classes(dataset)):
    
    inputs = keras.Input((image_size, image_size, 12))
    augmented = data_augmentation(inputs)
    
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(augmented)
    x = activation_block(x) 

    # Apply the mixer blocks
    for ii in range(depth):
        x = mixer_block(x, filters, kernel_size)

    # Attention Block to generate attention scores from the mixer_block output
    attention_output = linear_attention_map(x, filters)
    x = layers.Multiply()([x, attention_output])
    
    
    # Global Pooling and Final Classification Block
    x = layers.GlobalAvgPool2D()(x)  # Global pooling in 2D
    logits = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=[inputs], outputs=logits)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model

model= ConvAttentionNet(image_size=12, filters=96, depth=4, kernel_size=3, patch_size=2, num_classes=num_classes(dataset))
model.summary()

Aa = []
Oa = []
K = []

from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='accuracy', 
                              patience=10,
                              restore_best_weights=True
                              )

for i in range(10):
    print("Iteration number: ",i)
    
    
    # # Reset the model parameters at each iteration  
    model = ConvAttentionNet(image_size=windowSize, filters=96, depth=4, kernel_size=3, patch_size=2, num_classes=num_classes(dataset))
    
    history = model.fit(X_train, y_train,
                            batch_size = 64, 
                            verbose = 1, 
                            epochs = 100, 
                            shuffle = True,
                            callbacks = [early_stopper])
    
    #model.save_weights('./Models_Weights/'+ dataset +'/ConvAttentionNet_iter_' + str(i)+'.h5')
    #model.load_weights('./Models_Weights/'+ dataset +'/ConvAttentionNet_iter_' + str(i)+'.h5')

    Y_pred_test = predict_by_batching(model, X_test, X_test.shape[0]//64)
    #Y_pred_test = model.predict([X_test])
    y_pred_test = np.argmax(Y_pred_test, axis=1)
       
    
    
    
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1),  y_pred_test)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred_test)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    

    
   
    Aa.append(float(format((aa)*100, ".2f")))
    Oa.append(float(format((oa)*100, ".2f")))
    K.append(float(format((kappa)*100, ".2f")))
    
 

print("oa = ", Oa) 
print("aa = ", Aa)

print('Kappa = ', K)
print('\n')
print('Mean OA = ', format(np.mean(Oa), ".2f"), '+', format(np.std(Oa), ".2f"))
print('Mean AA = ', format(np.mean(Aa), ".2f"), '+', format(np.std(Aa), ".2f"))
print('Mean Kappa = ', format(np.mean(K), ".2f"), '+', format(np.std(K), ".2f"))
