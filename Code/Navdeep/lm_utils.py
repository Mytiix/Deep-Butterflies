# -*- coding: utf-8 -*-
"""
Created on Fri May 28 11:54:31 2021

@author: Navdeep Kumar
"""

import numpy as np
import pandas as pd
import os
import glob
import cv2 as cv
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import KFold

org_images = glob.glob('D:/BioMedAqu/BioMedAque_Datasets/GIGA/microscopic/images/*.bmp')
org_lmks = glob.glob('D:/BioMedAqu/BioMedAque_Datasets/GIGA/microscopic/landmarks/*.txt')

im_names = []
lmks_names = []
for i in range(len(lmks)):
    lm_name, _ = os.path.splitext(os.path.basename(lmks[i]))
    
    lmks_names.append(lm_name)

for j in range(len(images)):
    im_name, _ = os.path.splitext(os.path.basename(images[j]))
    im_names.append(im_name)

li = [item for item in im_names if item not in lmks_names]
li_lm = [item for item in lmks_names if item not in im_names]
os.chdir('D:/lm_test/lm_baseline/org_datasets/utv/xrays/renamed/landmarks')   
dest = 'D:/lm_test/lm_baseline/org_datasets/utv/xrays/renamed'

for file in li:
    filename  = file + '.txt'
    shutil.move(filename, dest) 

for i in range(len(org_lmks)):
    img = cv.imread(org_images[i], cv.IMREAD_UNCHANGED)
    lm = np.loadtxt(org_lmks[i])
    re_img, re_lm = rescale_pad_img(img, lm, 256)
    cv.imwrite('D:/lm_test/lm_baseline/org_datasets/giga/old113/images/'+str(i+1).zfill(3)+'.png', re_img)
    np.savetxt('D:/lm_test/lm_baseline/org_datasets/giga/old113/landmarks/'+str(i+1).zfill(3)+'.txt', re_lm, fmt='%d')


for i in range(len(images)):
    img = cv.imread(images[i],0)
    lm = np.loadtxt(lmks[i])
    plt.imshow(img)
    plt.scatter(lm[:,0], lm[:,1], c='r')
    plt.show()
    plt.savefig('D:/lm_test/plots_utmi/micro/'+str(i).zfill(3)+'.png')
    plt.clf()    
#=============== Rescale and Pad images landmarks =============================
def rescale_pad_img(image, landmarks, desired_size):
    
    h, w = image.shape[:2]
    
    aspect = w/h
    
    if aspect > 1 : #horizontal image
        new_w = desired_size
        new_h = int(desired_size*h/w)
        offset = int(new_w - new_h)
        if offset %  2 != 0: #odd offset
            top = offset//2 + 1
            bottom = offset//2
        else:
            top = bottom = offset//2
        
        dim = (new_w, new_h)
        re_img = cv.resize(image, dim, interpolation = cv.INTER_AREA)
        pad_img = cv.copyMakeBorder(re_img, top, bottom, 0,0, cv.BORDER_REPLICATE)
        x = landmarks[:,0]
        y = landmarks[:,1]
        new_x = x * new_w / w
        new_x = new_x.astype(int)
        new_y = y * new_h / h + offset//2
        new_y = new_y.astype(int)
            
    elif aspect < 1:  #vertical image
        new_h = desired_size
        new_w = int(desired_size*w/h)
        offset = int(np.ceil((new_h - new_w) // 2))
        if offset %  2 != 0: #odd offset
            top = offset -1
            bottom = offset
        else:
            top = bottom = offset
        dim = (new_w, new_h)
        re_img = cv.resize(image, dim, interpolation = cv.INTER_AREA)
        pad_img = cv.copyMakeBorder(re_img, top, bottom, 0,0, cv.BORDER_CONSTANT, value=0)
        new_x = x * new_w / w + offset//2
        new_x = new_x.astype(int)
        new_y = y * new_h / h
        new_y = new_y.astype(int)
    
    return pad_img, np.vstack((new_x, new_y)).T
#==============================================================================
for i in range(len(images)):
    im = cv.imread(images[i], 1) #grayscale image
    lm = np.loadtxt(lmks[i])
    
    re_img, re_lm = rescale_pad_img(im, lm, 256)
    cv.imwrite('D:/BioMedAqu/BioMedAque_Datasets/UTV/renamed/xrays/latest_rescaled/images/xrays_'+str(i)+'.jpg', re_img)
    np.savetxt('D:/BioMedAqu/BioMedAque_Datasets/UTV/renamed/xrays/latest_rescaled/landmarks/xrays_'+str(i)+'.txt', re_lm, fmt='%d')

idxs = [i for i in range(len(images))]
splits = [idxs[i:i + 10] for i in range(0, len(idxs), 10)] #10 is the number of element in each split
for j in range(len(splits)):
    train_idxs = [*splits[i], *splits[i+1], *split[i+2]]
    val_idxs = split[i+3]

fold = 1 
kf = KFold(n_splits=5)   
    
for train_idx, test_idx in kf.split(image_paths, lm_paths):
    #train_images = [images[i] for i in train_idx] # load images and masks from idx from k-fold split
    #train_lmks = [lmks[i] for i in train_idx]

    test_images = [image_paths[i] for i in test_idx]
    test_lmks = [lm_paths[i] for i in test_idx]
    
    tr_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=0)
    
    print(fold)
    print(X_train, y_train, X_val, y_val)
    fold = fold+1
        
def _checkBoundaries(p, target_size):
    if p[0] < 0:
            px = 0
    elif p[0] > target_size[0]:
            px = target_size[0]
    else:
            px = p[0]

        # y dimension
    if p[1] < 0:
            py = 0
    elif p[1] > target_size[1]:
            py = target_size[1]
    else:
            py = p[1]

    return (int(px), int(py))

def transforms(img, lm, transform_dict=None):
    aug_lm = []

    c = (img.shape[0] // 2, img.shape[1] // 2)
    if transform_dict['Flip']:
            flip = random.choice([True, False])
            if flip:
                img = np.fliplr(img)
    if transform_dict['Rotate']:

            if transform_dict['Scale']:
                s = random.uniform(0.7, 1.0)
            else:
                s = 1.0

            r = random.randint(-10, 10)
            M_rot = cv.getRotationMatrix2D(center=(img.shape[0] // 2, img.shape[1] // 2), angle=r, scale=s)
            img = cv.warpAffine(img, M_rot, (img.shape[0], img.shape[1]), borderMode=cv.BORDER_CONSTANT, borderValue=1)
    if transform_dict['Shift']:
            tx = random.randint(-20, 20)
            ty = random.randint(-20, 20)
            M_shift = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
            img = cv.warpAffine(img, M_shift, (img.shape[0], img.shape[1]),
                                borderMode=cv.BORDER_CONSTANT, borderValue=1)
            
    # transform keypoints
    c = (img.shape[0] // 2, img.shape[1] // 2)
        
    for i in range(0, len(lm) - 1, 2):

            px = lm[i,0]
            py = lm[i,1]
            p = np.array([px, py, 1], dtype=int)

            # apply flip
            if transform_dict['Flip'] and flip:
                p[0] = c[0] - (p[0] - c[0])

            # apply rotation
            if transform_dict['Rotate']:
                p = np.dot(M_rot, p)

            # apply horizontal / vertical shifts
            if transform_dict['Shift']:
                p[0] += tx
                p[1] += ty

            p = _checkBoundaries(p)

            aug_lm.append(p[0])
            aug_lm.append(p[1])
            
    return img, aug_lm

imgs = glob.glob('D:/lm_test/lm_baseline/dataset_giga_new/in/*.jpg')
lms = glob.glob('D:/lm_test/lm_baseline/dataset_giga_new/in/txt/*.txt')

li_lms = []
li_imgs = []

for i in range(len(lmks)):
    filename = os.path.basename(lmks[i])
    file, _ = os.path.splitext(filename)
    li_lms.append(file)
    
for j in range(len(images)):
    filen = os.path.basename(images[j])
    file, _ = os.path.splitext(filen)
    li_imgs.append(file)
    
rest = [item for item in li_imgs if item not in li_lms] 

source = 'D:/lm_test/lm_baseline/dataset_new/in/txt/'
os.chdir(source)
dest =  'D:/lm_test/lm_baseline/dataset_new/lateral/landmarks/'
for file in rest:
    print(file)
    filename = file+'.txt'
    shutil.move(filename, dest)
    
keypoints = glob.glob('D:/BioMedAqu/phd/DataSet/zebra_dataset113/org_dataset/zebra_txt/*.txt')    
for i in range(len(keypoints)):
    
    kp =  np.loadtxt(keypoints[i])
    lm_list = []
    
    lm_list.append(kp[30])
    lm_list.append(kp[12])
    lm_list.append(kp[17])
    lm_list.append(kp[16])
    lm_list.append(kp[4])
    lm_list.append(kp[15])
    lm_list.append(kp[6])
    lm_list.append(kp[11])
    lm_list.append(kp[3])
    lm_list.append(kp[8])
    lm_list.append(kp[0])
    lm_list.append(kp[2])
    lm_list.append(kp[24])
    lm_list.append(kp[7])
    lm_list.append(kp[14])
    lm_list.append(kp[21])
    lm_list.append(kp[18])
    lm_list.append(kp[25])
    lm_list.append(kp[28])
    lm_list.append(kp[26])
    lm_list.append(kp[9])
    lm_list.append(kp[23])
    lm_list.append(kp[22])
    lm_list.append(kp[5])
    lm_list.append(kp[1])
    lm_list.append(kp[27])
    lm_list.append(kp[29])
    lm_list.append(kp[20])
    lm_list.append(kp[19])
    lm_list.append(kp[10])
    lm_list.append(kp[13]) 
    lm_arr = np.array(lm_list)
    np.savetxt('D:/BioMedAqu/phd/DataSet/zebra_dataset113/org_dataset/new_landmarks/'+str(i+1).zfill(3)+'.txt', lm_arr, fmt='%d')
    
    
images = glob.glob('D:/lm_test/lm_baseline/dataset_xrays/in/*.jpg')
    
def crop_to_aspect(image, lm, asp_ratio, upsize):
    cr_h = image.shape[0] * asp_ratio
    cr_h = int(cr_h)
    cr_w = image.shape[1]
    crop_image = tf.image.resize_with_crop_or_pad(image, cr_h, cr_w)
    
    h = crop_image.shape[0]
    w = crop_image.shape[1]
    
    up_w = upsize
    up_h = int(upsize * h/w)
    
    up_image = tf.image.resize(crop_image, (up_h,up_w), method=ResizeMethod.AREA)
    
    offset = w - h
    x_lm = lm[:,0]
    y_lm = lm[:,1]
    
    y_lm = y_lm - offset//2  #height is reduced
    
    up_y_lm = y_lm * up_h / h
    up_x_lm = x_lm * up_w / w
    
    return up_image, np.vstack((up_x_lm, up_y_lm)).T
    
h = org_img.shape[0] 
w = org_img.shape[1]  

test_images = [org_image_paths[i] for i in test_idxs]
test_lmks = [org_lm_paths[i] for i in test_idxs]
re_images = [image_paths[i] for i in test_idxs]

image = cv.imread(re_images[85])
pred_lm = pred_landmarks[85]
gt_lm = gt_landmarks[85]


up_lm = up_landmarks[70]
org_img = cv.imread(org_image_paths[70]) 
org_img = cv.cvtColor(org_img, cv.COLOR_BGR2RGB)
org_lm = np.loadtxt(org_lm_paths[70])
img = tf.image.resize_with_pad(org_img, 256,256, method=tf.image.ResizeMethod.AREA)
img = img.numpy()
img = np.float32(img) / 255
img = np.expand_dims(img, axis=0)
pred_mask = model.predict(img)
pred_mask = np.reshape(pred_mask, newshape=(256,256, 31))
nChannels = pred_mask.shape[-1]
pred_list = []
for k in range(nChannels):
        xpred, ypred = maskToKeypoints(pred_mask[:, :, k])
        pred_list.append((xpred, ypred))
        #pred_list.append((xpred, ypred))
        pred_lmks = np.array(pred_list)

x_lm = pred_lmks[:,0]
y_lm = pred_lmks[:,1]
y_lm = y_lm - offset//2

up_y_lm = y_lm * up_h / h
up_x_lm = x_lm * up_w / w

plt.imshow(org_img)
plt.scatter(org_lm[:,0], org_lm[:,1], c='b') 
plt.scatter(up_lm[:,0], up_lm[:,1], c='r')

plt.imshow(image)
plt.scatter(gt_lm[:,0], gt_lm[:,1], c='b') 
plt.scatter(pred_lm[:,0], pred_lm[:,1], c='r')

lm_path = glob.glob('D:/lm_test/lm_baseline/org_datasets/utv/xrays/renamed/rescaled/landmarks/*.txt')
org_lm_paths = glob.glob('D:/lm_test/lm_baseline/org_datasets/utv/xrays/renamed/landmarks/*.txt') 
l_lm = []    
for i in range(len(lm_path)):
    lm = np.loadtxt(lm_path[i])
    n = len(lm)
    l_lm.append(n)
    
lmks_names = []    
for i in range(len(org_lm_paths)):
    lm_name, _ = os.path.splitext(os.path.basename(org_lm_paths[i]))
    
    lmks_names.append(lm_name)
    
re_lmks_names = []    
for i in range(len(lm_path)):
    lm_name, _ = os.path.splitext(os.path.basename(lm_path[i]))
    
    re_lmks_names.append(lm_name)
    
#============= resizing nd padding a single image=============================
def rescale_pad_img(image, desired_size):
    
    h, w = image.shape[:2]
    
    aspect = w/h
    
    if aspect > 1 : #horizontal image
        new_w = desired_size
        new_h = int(desired_size*h/w)
        offset = int(new_w - new_h)
        if offset %  2 != 0: #odd offset
            top = offset//2 + 1
            bottom = offset//2
        else:
            top = bottom = offset//2
        
        dim = (new_w, new_h)
        re_img = cv.resize(image, dim, interpolation = cv.INTER_AREA)
        pad_img = cv.copyMakeBorder(re_img, top, bottom, 0,0, cv.BORDER_REPLICATE)
            
    elif aspect < 1:  #vertical image
        new_h = desired_size
        new_w = int(desired_size*w/h)
        offset = int(np.ceil((new_h - new_w) // 2))
        if offset %  2 != 0: #odd offset
            top = offset -1
            bottom = offset
        else:
            top = bottom = offset
        dim = (new_w, new_h)
        re_img = cv.resize(image, dim, interpolation = cv.INTER_AREA)
        pad_img = cv.copyMakeBorder(re_img, top, bottom, 0,0, cv.BORDER_REPLICATE)
    
    return pad_img

image_paths = glob.glob('D:/lm_test/alessio_pics/*')

for i in range(len(image_paths)):
    img = cv.imread(image_paths[i],0)
    filename = os.path.basename(image_paths[i])
    file, ext = os.path.splitext(filename)
    re_img = rescale_pad_img(img, 256)
    cv.imwrite('D:/lm_test/alessio_renamed/'+file+'.png', re_img)