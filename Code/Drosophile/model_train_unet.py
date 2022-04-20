# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:46:30 2021

@author: Navdeep Kumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import glob
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras import utils
from data_generator import *
from landmark_models import *
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())

def show_masks(batch_imgs, batch_gt_masks, nrows, ncols, include_preds= False, predictions=None):

    if not include_preds:
        nrows -= 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    #r = -1

    for c in range(ncols):

        # original image
        img = batch_imgs[c]
        img = np.reshape(img, newshape=(256,256,3))
        #img = np.stack([img,img,img], axis=-1)

        # ground-truth mask
        gt_mask = batch_gt_masks[c]
        gt_mask = np.reshape(gt_mask, newshape=(256,256,31))
        gt_mask = np.sum(gt_mask, axis=-1)

        axes[0, c].imshow(img)
        axes[1, c].imshow(gt_mask)

        # prediction mask
        if include_preds: 
            pred_mask = predictions[c]
            pred_mask = np.reshape(pred_mask, newshape=(256,256, 31))
            pred_mask = np.sum(pred_mask, axis=-1)
            axes[2, c].imshow(pred_mask)

    plt.show()
    
#========== Load Train data ===================================================

root = './landmark_dataset/giga/old_micro'

images = glob.glob(os.path.join(root,'images/*.png'))
keypoints = glob.glob(os.path.join(root,'landmarks/*.txt'))

images = os.listdir(os.path.join(root,'images'))

idxs = []
img_dict = {}
kp_dict = {}

for i in range(len(images)):
    idxs.append(i)
    
    img_dict[i] = images[i]
    
    kp = np.loadtxt(keypoints[i], delimiter=' ')
    kp = np.array(kp).astype(np.int32)
    kp = kp.flatten('C')
    kp = list(kp)
    
    
    kp_dict[i] = kp
#==============================================================================
random.shuffle(idxs)

# subset = int(0.1*len(idxs))

cutoff_idx = int(0.9*len(idxs))
train_idxs = idxs[0:cutoff_idx]
val_idxs = idxs[cutoff_idx:len(idxs)]

print("\n# of Training Images: {}".format(len(train_idxs)))
print("# of Val Images: {}".format(len(val_idxs)))

transform_dict = {"Flip": True, "Shift": True, "Scale": True, "Rotate": True}

train_gen = MaskGenerator(os.path.join(root, 'images'),
                              train_idxs,
                              img_dict,
                              kp_dict,
                              transform_dict=transform_dict,
                              augment=False, 
                              batch_size=4)

val_gen = MaskGenerator(os.path.join(root, 'images'),
                            val_idxs,
                            img_dict,
                            kp_dict,
                            augment=False,
                            batch_size=4)

print("\n# of training batches= %d" % len(train_gen))
print("# of validation batches= %d" % len(val_gen))

train_imgs, train_masks = train_gen[0]
print(train_imgs.shape)
print(train_masks.shape)
#show_masks(train_imgs[0:4], train_masks[0:4], nrows=3, ncols=4)

val_imgs, val_masks= val_gen[0]
print(val_imgs.shape)
print(val_masks.shape)
#show_masks(val_imgs[0:4], val_masks[0:4], nrows=3, ncols=4)

#================ Training ====================================================
def jaccard(ytrue, ypred, smooth=1e-5):

    intersection = K.sum(K.abs(ytrue*ypred), axis=-1)
    union = K.sum(K.abs(ytrue)+K.abs(ypred), axis=-1)
    jac = (intersection + smooth) / (union-intersection+smooth)

    return K.mean(jac)
def mean_squared_error(y_true, y_pred):
    channel_loss = K.sum(K.square(y_pred - y_true), axis=-1)
    total_loss = K.mean(channel_loss, axis=-1)
    print(total_loss.shape)
    return total_loss


#==============================================================================
    
steps_per_epoch = len(train_gen)
val_steps = len(val_gen)
def trainModel(model, model_name, loss_type, n_epochs,lr):
    filepath="./saved_models/"+model_name+"/weights-improvement-{epoch:02d}-{val_loss:.8f}.hdf5"
    log_dir="logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=150)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=0.001)
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                                  write_images=False)
    callbacks = [checkpoint, tensor_board, early_stopping, reduce_lr]
    
    optim = RMSprop(lr=lr)
    rmse = tf.keras.metrics.RootMeanSquaredError()
    model.compile(loss="mse", optimizer=optim, metrics=[rmse])
    model.fit(train_gen, epochs=n_epochs, callbacks=callbacks,
                        validation_data=val_gen, steps_per_epoch= steps_per_epoch,
                        validation_steps= val_steps,verbose=1, workers=6, use_multiprocessing=True)

    return model
#==============================================================================
    
loss_type = "mse"

model = UNET(input_shape=(256, 256, 3))

history = trainModel(model, "unet", loss_type, n_epochs=10, lr=1e-3)

        