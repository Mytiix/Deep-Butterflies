# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:45:17 2022

@author: Navdeep Kumar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import glob
import tensorflow as tf
#import tensorflow.keras.backend as K
import cv2 as cv
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras import utils
from data_generator import *
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model, load_model


#model = load_model('./figures/saved_models/giga/fcn/fcn1.hdf5')
model = load_model('D:/lm_test/updated_saved_models/model_ds/utv/microscopic/exp/unet/weights-improvement-138-0.000052922056.hdf5')
#model = load_model('./updated_saved_models/utv/xrays/unet/weights-improvement-360-0.00007002.hdf5')
#root = 'D:/lm_test/lm_baseline/org_datasets/giga/old113/test_set'
#root = 'D:/lm_test/lm_baseline/org_datasets/utv/microscopic/renamed/rescaled/test_set'
#root = 'D:/BioMedAqu/phd/DataSet/zebra_dataset113/rescaled256'
test_images = glob.glob('D:/lm_test/lm_baseline/org_datasets/utv/xrays/renamed/rescaled/test_set/images/*.png')
test_lmks = glob.glob('D:/lm_test/lm_baseline/org_datasets/utv/xrays/renamed/rescaled/test_set/landmarks/*.txt')

org_images = glob.glob('D:/lm_test/lm_baseline/org_datasets/utv/xrays/renamed/images/*')
org_lmks = glob.glob('D:/lm_test/lm_baseline/org_datasets/utv/xrays/renamed/landmarks/*')


idxs = []
kp_list = []
img_dict = {}
for i in range(len(test_images)):
    idxs.append(i)
    
    #img_dict[i] = image_paths[i]
    
    kp = np.loadtxt(test_lmks[i], delimiter=' ')
    kp = np.array(kp).astype(np.int32)
    #kp = kp.flatten('C')
    kp_list.append(kp)

#============== hyperparameters ===============================================
image_size = (256,256,1)
channels = [3,1]
N = 25
batch_size = 4
#================ Mapping functions ===========================================

def parse_data(image, lm):
    """
    Reads image and landmark files depending on
    specified extension.
    """
    image_content = tf.io.read_file(image)

    image = tf.io.decode_png(image_content, channels=channels[1])
    image = tf.image.resize(image, (256,256))
    
    #image = tf.image.resize_with_pad(image, 256,256)
    
    return  image, lm

def normalize(image, lm):
    image = tf.cast(image, tf.float32)/255.0
    return image, lm

    #========= some utils ========================================================
def maskToKeypoints(mask):
    # mask = np.reshape(mask, newshape=(96,96))
    kp = np.unravel_index(np.argmax(mask, axis=None), shape=(256,256))
    return kp[1], kp[0]

def up_lm(lmks,curr_size, upsize):
    asp_ratio = upsize[0]/upsize[1] #h/w
    
    w = curr_size
    h = w * asp_ratio
    
    up_w = upsize[1]
    up_h = upsize[0]
    
    offset = w - h
    x_lm = lmks[:,0]
    y_lm = lmks[:,1]
    
    y_lm = y_lm - offset//2  #height is reduced
    
    up_y_lm = y_lm * up_h / h
    up_x_lm = x_lm * up_w / w
    
    return np.vstack((up_x_lm, up_y_lm)).T
    
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_lmks))
test_ds = test_ds.map(parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(batch_size)

preds = model.predict(test_ds)

org_landmarks = []
pred_landmarks= []

for i in range(len(test_images)):
    org_img = cv.imread(org_images[i])
    org_lm = np.loadtxt(org_lmks[i])
    pred_mask = preds[i]
    pred_mask = np.reshape(pred_mask, newshape=(256,256, N))
    lm_list = []
    for j in range(N):
        x,y =  maskToKeypoints(pred_mask[:, :, j])
        lm_list.append((x,y))
    pred_lm = np.array(lm_list)
    up_size = org_img.shape[:2]
    up_lmks = up_lm(pred_lm, 256, up_size)
    org_landmarks.append(org_lm)
    pred_landmarks.append(up_lmks)
error = []
for k in range(len(org_landmarks)):
    g = org_landmarks[k]
    p = pred_landmarks[k]
    mse = mean_squared_error(g, p, squared=False)
    error.append(mse)
mean_error = np.array(error).mean()
print(mean_error)

per_lm_distance = []
for i in range(len(org_landmarks)):
    euc_dist = []
    for j in range(N):
        g_lm = org_landmarks[i][j]
        p_lm = pred_landmarks[i][j]
        
        dist = np.linalg.norm(g_lm - p_lm)
        euc_dist.append(dist)
    per_lm_distance.append(np.array(euc_dist))
    per_lm_final = np.array(per_lm_distance)
mean_dist = np.mean(per_lm_final, axis=0)
np.savetxt('D:/lm_test/lm_baseline/per_lm_errors/G_xr_fcn.txt', mean_dist, fmt='%d')
print(np.mean(mean_dist))
        
        
        

for i in range(len(org_images)):
    img = cv.imread(org_images[11])
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    gt_lm = org_landmarks[11]
    pred_lm = pred_landmarks[11]
    
    plt.figure()
    plt.subplot(1, 2,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original Image')
    
    plt.subplot(1,2,2)
    plt.title('Predicted (red dots) and ground truth landmarks (blue dots)')
    plt.imshow(img)
    plt.axis('off')
    plt.scatter(gt_lm[:,0], gt_lm[:,1], c= 'b')
    plt.scatter(pred_lm[:,0], pred_lm[:,1], c= 'r')
    plt.show()

#============ With single image =============================================
image_path = glob.glob('D:/lm_test/newGIGA_testImages/*')
org_img = cv.imread(image_path[1])
org_img = cv.cvtColor(org_img, cv.COLOR_BGR2RGB)
img = rescale_pad_img(org_img, 256)
norm_img = img/255.0
norm_img = np.expand_dims(norm_img, axis=0)
pred = model.predict(norm_img)
pred = np.reshape(pred, newshape=(256,256, N))
lm_list = []
for j in range(N):
    x,y =  maskToKeypoints(pred[:, :, j])
    lm_list.append((x,y))
pred_lm = np.array(lm_list)
up_size = org_img.shape[:2]
up_lmk = up_lm(pred_lm, 256, up_size)

plt.imshow(org_img)
plt.scatter(up_lmk[:,0], up_lmk[:,1], c='red')


#=============================================================================

for i in range(20):
    plt.figure()
    img = cv.imread(org_images[1])
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    gt_lm = np.loadtxt(org_lmks[1])

    plt.imshow(img)
    plt.scatter(gt_lm[:,0], gt_lm[:,1],s=10, c= 'b')
    plt.show()