#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:22:40 2020

@author: maywu
"""

import os
from read_roi import read_roi_zip, read_roi_file
#from helperfunctions import *
from skimage.draw import polygon2mask
import numpy as np
from bresenham import bresenham
import matplotlib.pyplot as plt
import os
import re
import h5py
#%%
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import random
from read_datasets import get_img_roi_dirs


def data_aug(img, lb, i):
    img = Image.fromarray(img)
    lb = Image.fromarray(lb)
    ranges = {        
                "contrast": np.linspace(0.0, 0.9, 10),           
                "brightness": np.linspace(0.0, 0.9, 10),
            }    
 
    n = i%3
    a = random.choice([-1, 1]) 
    if  n == 2:
        enhancer = ImageEnhance.Brightness(img)
        result = enhancer.enhance( 1+ranges['brightness'][5] * a)
        result2 = lb
        
    elif n == 1: 
        enhancer = ImageEnhance.Contrast(img)
        result = enhancer.enhance( 1+ranges['contrast'][5] * a)
        result2 = lb
    elif n == 0:
        result = np.fliplr(np.asarray(img))
        result2 = np.fliplr(np.asarray(lb))
        
    return np.asarray(result), np.asarray(result2)

        

#%%
from pathlib import Path
data_rootpath = 'Y:/skala/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/0-segmentation_completed'

roi_dirs = [] # list holding all roi dir
img_dirs = [] # list for all images dir

keywords= ['10_10'] # keywords used to search files
img_dirs, roi_dirs = get_img_roi_dirs(data_rootpath, keywords)
height = 265
width = 3100    
num_class = 5  
background = 5  
    
labels = np.ones((height, width, 2, len(roi_dirs)*2))* num_class
images = np.zeros((height, width, 1, len(roi_dirs)*2))
l =  len(roi_dirs)*2
i = 0
################## debug no such file or dir error
#test = 'Y:\\skala\\0-Projects and Experiments\\KS - OCT membranes\\oct_dataset_3100x256\\0-segmentation_completed\\2018_10_09_human_amniochorion_labored_term_AROM_periplacental_0002_Mode2D\\images\\2018_10_09_human_amniochorion_labored_term_AROM_periplacental_0002_Mode2D_0.tiff'
#plt.imread(test)
####################
for dirs in roi_dirs:
    #print(dirs)
    img = plt.imread(img_dirs[i]) # couldn't read in files???????
    dim = img.shape
    if len(dim) > 2:
        img = img[:,:,0]
    images[:,:,0,i] = img
    labels[:,:,0,i] = get_label(dirs, num_class, width, height)
    result_img , result_lb = data_aug(img, labels[:,:,0,i], i)
    images[:,:,0,l-1-i] = result_img # save the augmented data at the end of the matrix
    labels[:,:,0,l-1-i] = result_lb
    i += 1
w = weight(labels[:,:,0,:],3,6, background) # 3: weight for layer transition, 6: weight for background vs. placenta   
labels[:,:,1,:] = w

Set = np.ones((1,len(labels[0,0,0,:])))
Set[0,int(len(Set[0,:])*0.8):] = 3 # 1: training, 3 validation
# =============================================================================
# with h5py.File(data_rootpath + '/set_plac.h5', 'w') as f:
#     f.create_dataset('Set', data = Set) 
# with h5py.File(data_rootpath+'/Data_plac.h5', 'w') as f:
#     f.create_dataset('oct_dataset', data = images)  
# with h5py.File(data_rootpath+'/label_plac.h5', 'w') as f:
#     f.create_dataset('oct_labels', data = labels)  
# 
# =============================================================================
#%%
def get_label(directory, num_class = 5, width = 3100, height = 265):
    """
    read in roi files, creates the transition layer matrix (#row = #class, #col = width) from
    the x and y coordinates saved in the roi files, 
    and then create the mask image (width, height) from the transition layer matrix.

    Parameters
    ----------
    directory : string
        full path of the directory to the roi files.
    width : int, optional
        the width of the image. The default is 3100.
    height: int, optional
        the height of the image. the default is 265
    num_class: int, optional
        the number of rows of the transition layer matrix, the number of 
        placenta layer classes

    Returns
    -------
    lb : numpy array
        the mask created from the roi file. lb is an image.

    """
    # get roi info from a RoiSet.zip 
    roi = read_roi_zip(directory)      
    # a matrix for labels
    label = np.zeros((num_class, width)) #TODO: make them variables
    index_layer = 0
    for key in roi.keys():
        x_layer = roi[key]['x']
        y_layer = roi[key]['y']
        prev_x = x_layer[0]
        prev_y = y_layer[0]
        for x, y in zip(x_layer, y_layer):
            if x == x_layer or y == y_layer:
                continue
            else:
                line = list(bresenham(prev_x,prev_y,x,y))
                prev_coord_x = 0
                for coord in line:
                    if coord[0] == prev_coord_x:
                        continue
                    else:
                        label[index_layer][coord[0]] = coord[1]
                        prev_coord_x = coord[0]
                prev_x = x
                prev_y = y
        index_layer +=1           
    lb = create_labels(label, num_class, width, height)
    return lb
        

def label_a_col(label_for_col, height = 265, num_class = 5):
    """
    The height and num_classes might have to be hard coded here.
    not sure how to use map() function on mathods taking multiple parameters. 
    takes in a 1d list indicating the layer transition of a column and recreate the actual image label
    with size of the image
    the endpoints inclusive or exclusive?
    """
    col = np.ones(height) * num_class 
    for i in range(len(label_for_col)): 
        if label_for_col[i] == 0:
            continue
        col[int(label_for_col[i]):] = i + 1 
    return col
    

    
def create_labels(labels, num_class = 5, width = 3100, height = 265):
    """
    create labels for a 2d image
    """
    labels = np.swapaxes(labels,0,1)
    result = map(label_a_col, labels)
    output =[]
    for x in result:
        output.append(x)
    return np.swapaxes(output,0,1)


def image_show(image, nrows=1, ncols=1):  # , cmap='gray'

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    ax.imshow(image)  # , cmap ='gray'
    ax.axis('off')
    return fig, ax


def weight(labels, w1, w2, background = 5):
    """
    w(x) = 1 + w1 * I(|gradient(l(x))|) + w2 * I(l(x) = L)
    L = retinal layers and fluid masses
    :param labels: a list of labels
    :param w1:
    :param w2:
    :background: the number that represents background
    :return:
    """    
    weights = np.zeros((labels.shape)) 
    labels = np.array(labels)
    for i in range(labels.shape[2]):
        label = labels[:,:,i]
        I_w2 = label
        label = np.gradient(label)
        grad = list(map(abs, label))
        abs_grad = grad[0]+grad[1]
        abs_grad[abs_grad>0] = 1
        I_w1 = abs_grad
        pos1 = I_w2 == background # label of background
        #pos2 = I_w2 == 6 # this pos2 can be deleted
        I_w2 = 1*(~np.array(pos1 + pos2))
        weight = 1+ w1*I_w1 + w2*I_w2
        weights[:,:,i] = weight
        ####
        #plt.imshow(weights[:,:,i])
        #plt.show()
        #####
    return weights


 

