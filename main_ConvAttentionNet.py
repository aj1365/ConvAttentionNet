import keras
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

import numpy as np
from keras import layers
import tensorflow as tf
# %matplotlib inline

from SAR_utils import *




## GLOBAL VARIABLES
dataset = 'FL' # FL, SF, ober
test_ratio = 0.99


    


X , Y, Labels = loadData(dataset)

img_display(classes=Y, title='groundtruth', class_name=Labels)


X = (X-np.min(X))/(np.max(X)-np.min(X))



indexes, labels = get_img_indexes(Y, removeZeroindexes = True)


X_train_idx, X_test_idx, y_train, y_test = splitTrainTestSet(indexes, labels, testRatio = 0.99)

sample_report = f"{'class': ^25}{'train_num':^10}{'test_num': ^10}{'total': ^10}\n"
for i in np.unique(Y):
    if i == 0: continue
    sample_report += f"{Labels[i-1]: ^25}{(y_train==i-1).sum(): ^10}{(y_test==i-1).sum(): ^10}{(Y==i).sum(): ^10}\n"
sample_report += f"{'total': ^25}{len(y_train-1): ^10}{len(y_test-1): ^10}{len(labels): ^10}"
print(sample_report)

windowSize = 12
X_train = createImageCubes(X, X_train_idx, windowSize)
y_train = keras.utils.to_categorical(y_train)
    


###############################################################################
def linear_attention_map(inputs, out_channel):

    # First convolution with linear kernel size (6, 1) for horizontal filtering
    conv2 = layers.Conv2D(out_channel // 2, kernel_size=(6, 1), padding='same', use_bias=True)(inputs)
    act2 = layers.Activation("relu")(conv2)

    # Second convolution with linear kernel size (1, 6) for vertical filtering
    conv3 = layers.Conv2D(out_channel // 2, kernel_size=(1, 6), padding='same', use_bias=True)(act2)
    act3 = layers.Activation("relu")(conv3)

    # Final 1x1 convolution to restore the output channels
    conv4 = layers.Conv2D(out_channel, kernel_size=1, use_bias=True)(act3)
    attention_map = layers.Activation("sigmoid")(conv4)

    return attention_map

    

def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(windowSize, windowSize),
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
    x = activation_block(x)  # Ensure this is defined somewhere

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


from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='accuracy', 
                              patience=10,
                              restore_best_weights=True
                              )

    
history = model.fit(X_train, y_train,
                    batch_size = 64, 
                    verbose = 1, 
                    epochs = 100, 
                    shuffle = True,
                    callbacks = [early_stopper])
    
Y_pred_test = predict_by_batching(model, input_tensor_idx = X_test_idx, batch_size = 1000, X = X, windowSize = windowSize)
y_pred_test = np.argmax(Y_pred_test, axis=1)
       
    
    
    
kappa = cohen_kappa_score(y_test,  y_pred_test)
oa = accuracy_score(y_test, y_pred_test)
confusion = confusion_matrix(y_test, y_pred_test)
each_acc, aa = AA_andEachClassAccuracy(confusion)

print('Overall Accuracy = ', format(oa*100, ".2f"))
print('Average Accuracy = ', format(aa*100, ".2f"))
print('Kappa = ', format(kappa*100, ".2f"))


Predicted_Class_Map = get_class_map(model, X=X, label = Y, window_size = windowSize)
img_display(classes=Predicted_Class_Map, title='Predicted', class_name=Labels)

## To produce masked version of the predicted map
gt_binary = Y
gt_binary[Y>0]=1
img_display(classes=Predicted_Class_Map*gt_binary, title='Predicted_with Mask', class_name=Labels)

    

