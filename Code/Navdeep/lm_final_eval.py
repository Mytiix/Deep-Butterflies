# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:51:26 2021

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

N=14

#model = load_model('./figures/saved_models/giga/fcn/fcn1.hdf5')
model = load_model('./lm_scripts/saved_models/unet/unet1.hdf5')
#model = load_model('./updated_saved_models/utv/xrays/unet/weights-improvement-360-0.00007002.hdf5')
root = 'D:\\Dataset_TFE\\images_v2\\polyphemus\\v\\testing\\rescaled'
#root = 'D:/lm_test/lm_baseline/org_datasets/utv/microscopic/renamed/rescaled/test_set'
#root = 'D:/BioMedAqu/phd/DataSet/zebra_dataset113/rescaled256'
images = glob.glob(os.path.join(root,'images/*.png'))
keypoints = glob.glob(os.path.join(root,'landmarks/*.txt'))
image_paths = glob.glob(os.path.join(root,'images/*.png'))
images = os.listdir(os.path.join(root,'images'))

idxs = []
img_dict = {}
kp_dict = {}

for i in range(len(images)):
    idxs.append(i)
    
    img_dict[i] = images[i]
    
    kp = np.loadtxt(keypoints[i], delimiter=' ')
    #kp = np.delete(kp,[29,30],axis=0)
    kp = np.array(kp).astype(np.int32)
    kp = kp.flatten('C')
    kp = list(kp)
    
    
    kp_dict[i] = kp
    
#=============================================================================
seed= 44
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
indexes = []
for train_idxs, test_idxs in kfold.split(images, keypoints):
    #test_images = [images[i] for i in test_idxs]
    #test_lmks = [keypoints[i] for i in test_idxs]
    #(train, test) = train_idxs, test_idxs
    indexes.append((train_idxs, test_idxs))
    print("\n# of Training Images: {}".format(len(train_idxs)))
    print("# of Val Images: {}".format(len(test_idxs)))
test_idxs = indexes[0][1]      
test_gen = MaskGenerator(os.path.join(root, 'images'),
                            idxs,
                            img_dict,
                            kp_dict,
                            augment=False,
                            batch_size=1,
                            shuffle=False)


#========== from heatmap to landmarks =========================================
def maskToKeypoints(mask):
    # mask = np.reshape(mask, newshape=(96,96))
    kp = np.unravel_index(np.argmax(mask, axis=None), shape=(256,256))
    return kp[1], kp[0]

#imgs, lmks = test_gen[1]
#==============================================================================


gt_landmarks = []
pred_landmarks = []
for i in range(len(test_gen)):
    imgs, masks = test_gen[i]
    preds = model.predict(imgs)
    
    for j in range(len(imgs)):
        image = imgs[j]
        gt_mask = masks[j]
        pred_mask = preds[j]
        
        gt_mask = np.reshape(gt_mask, newshape=(256,256, N))
        pred_mask = np.reshape(pred_mask, newshape=(256,256, N))
        
        gt_list = []
        pred_list = []
        for k in range(N):
            xgt, ygt = maskToKeypoints(gt_mask[:, :, k])
            xpred, ypred = maskToKeypoints(pred_mask[:, :, k])
            
            gt_list.append((xgt, ygt))
            pred_list.append((xpred, ypred))
            
        
        #gt_list.sort(key=lambda tup: (tup[0], tup[1]))
        gt_lmks = np.array(gt_list)
        gt_landmarks.append(gt_lmks)
        
        #pred_list.sort(key=lambda tup: (tup[0],tup[1]))
        pred_lmks = np.array(pred_list)
        pred_landmarks.append(pred_lmks)

#========== upscale landmarks to original scale ==============================
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

org_image_paths = glob.glob('D:\\Dataset_TFE\\images_v2\\polyphemus\\v/testing/images/*.tif')
org_lm_paths = glob.glob('D:\\Dataset_TFE\\images_v2\\polyphemus\\v/testing/landmarks/*.txt')

#test_images = [org_image_paths[a] for a in test_idxs]
#test_lmks = [org_lm_paths[a] for a in test_idxs]
#re_images = [image_paths[i] for i in test_idxs]

up_landmarks = []
for i in range(len(pred_landmarks)):
    lmks = pred_landmarks[i]
    org_img = cv.imread(org_image_paths[i])
    upsize = org_img.shape[:2]
    up_lmks = up_lm(lmks, 256, upsize)
    
    up_landmarks.append(up_lmks)

#org_landmarks = glob.glob('D:/lm_test/lm_baseline/org_datasets/utv/xrays/renamed/landmarks/*.txt')
#org_test_landmarks = [org_landmarks[i] for i in test_idxs]
org_gt_landmarks= []
for i in range(len(org_lm_paths)):
    lmks = np.loadtxt(org_lm_paths[i])
    lmks = lmks.astype(np.int32) 
    org_gt_landmarks.append(lmks) 

    
#mse = (np.square(gt_lmks - pred_lmks)).mean(axis=None)

org_error = []
for i in range(len(org_gt_landmarks)):
    g = org_gt_landmarks[i]
    p = up_landmarks[i]
    mse = mean_squared_error(g, p, squared=False)
    #mse = np.sqrt(np.square(g-p).mean(axis=None))
    
    org_error.append(mse)


    
mean_error = np.array(org_error).mean()
print(mean_error)

for i in range(len(org_image_paths)): 
    img_test = cv.imread(org_image_paths[i])
    img_test = cv.cvtColor(img_test, cv.COLOR_BGR2RGB)
    lm = np.loadtxt(org_lm_paths[i])
    up_lm = up_landmarks[i]
    plt.figure()
    plt.imshow(img_test)
    plt.scatter(lm[:,0],lm[:,1], c='b')
    plt.scatter(up_lm[:,0],up_lm[:,1], c='r')
    plt.show()


plt.clf()
img_test = cv.imread(image_paths[30])
lm = np.loadtxt(keypoints[30])
pred_lm = pred_landmarks[30]
plt.imshow(img_test)
plt.scatter(lm[:,0],lm[:,1], c='b')
plt.scatter(pred_lm[:,0], pred_lm[:,1], c='r')


pred_arr = []
gt_arr = []
for i in range(len(org_gt_landmarks)):
    gt_arr = org_gt_landmarks[i]
    pred_arr = up_landmarks[i]
    #gt_arr = np.delete(gt_arr,[29,30],axis=0)
    #pred_arr = np.delete(pred_arr,[29,30],axis=0)
    
    gt_final.append(gt_arr)
    pred_final.append(pred_arr)

error = []
for i in range(len(gt_landmarks)):
    g = gt_landmarks[i]
    p = pred_landmarks[i]
    mse = mean_squared_error(g, p, squared=False)
    #mse = np.sqrt(np.square(g-p).mean(axis=None))
    
    error.append(mse)


    
mean_error = np.array(error).mean()
print(mean_error)     

plt.clf()
img_test = cv.imread(org_image_paths[4])
img_test = cv.cvtColor(img_test,cv.COLOR_BGR2RGB)
lm = gt_final[4]
pred_lm = pred_final[4]
plt.imshow(img_test)
plt.scatter(lm[:,0],lm[:,1], c='r')
plt.scatter(pred_lm[:,0], pred_lm[:,1], c='r')  

barWidth = 0.25
fig = plt.subplots()
br1 = np.arange(len(XRAYS))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
 
# Make the plot
h1 = plt.bar(br1, XRAYS, color ='r', width = barWidth,
        edgecolor ='grey', label ='XRAYS')
h2 = plt.bar(br2, UTV_MICRO, color ='g', width = barWidth,
        edgecolor ='grey', label ='UTV_MICRO')
h3 = plt.bar(br3, GIGA_MICRO, color ='b', width = barWidth,
        edgecolor ='grey', label ='GIGA_MICRO')
 
# Adding Xticks
plt.xlabel('Datasets', fontweight ='bold', fontsize = 15)
plt.ylabel('MSE Error (in pixels)', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(XRAYS))],
        ['UNET', 'FCN8'])
plt.legend(['XRAYS', 'UTV_MICRO', 'GIGA_MICRO'])

for rect in h1 + h2 + h3:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')

plt.legend()
#plt.tight_layout()
plt.show()
 
plt.legend()
plt.show()  

