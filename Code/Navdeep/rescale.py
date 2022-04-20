import numpy as np
import cv2 as cv
import glob
import os

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


if __name__ == '__main__':
	repository = 'D:\\Dataset_TFE\\images_v2\\hercules_telemachus_amphitryon\\d\\training'

	org_images = glob.glob(repository+'/images/*.tif')
	org_lmks = glob.glob(repository+'/landmarks/*.txt')

	if not os.path.exists(repository+'/rescaled'):
		os.makedirs(repository+'/rescaled')
		os.makedirs(repository+'/rescaled/images')
		os.makedirs(repository+'/rescaled/landmarks')


	for i in range(len(org_lmks)):
	    img = cv.imread(org_images[i], cv.IMREAD_UNCHANGED)
	    lm = np.loadtxt(org_lmks[i])
	    #lm = lm[:2]
	    re_img, re_lm = rescale_pad_img(img, lm, 256)
	    cv.imwrite(repository+'/rescaled/images/'+str(i+1).zfill(3)+'.png', re_img)
	    np.savetxt(repository+'/rescaled/landmarks/'+str(i+1).zfill(3)+'.txt', re_lm, fmt='%d')
