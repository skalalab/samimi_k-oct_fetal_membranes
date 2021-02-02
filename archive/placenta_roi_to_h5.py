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


def data_aug(img, lb, i):
    img = Image.fromarray(img)
    lb = Image.fromarray(lb)
    ranges = {
                "shearX": np.linspace(0, 0.3, 10),
                "shearY": np.linspace(0, 0.3, 10),            
                "rotate": np.linspace(0, 30, 10),            
                "contrast": np.linspace(0.0, 0.9, 10),           
                "brightness": np.linspace(0.0, 0.9, 10),
            }    
 
    n = i%6
    a = random.choice([-1, 1]) 
    '''
    if n == 7:
        result = img.transform(
        img.size, Image.AFFINE, (1, ranges['shearX'][9]* a, 0, 0, 1, 0), Image.BICUBIC, fillcolor=0)
        
        result2 = lb.transform(
        img.size, Image.AFFINE, (1, ranges['shearX'][9]* a, 0, 0, 1, 0), Image.BICUBIC, fillcolor=5)

    elif n == 7:
        im2 = img.convert("RGBA")
        rot = im2.rotate(ranges['rotate'][3]*a)
        fff = Image.new("RGBA", rot.size, (0,) * 4)
        out = Image.composite(rot, fff, rot)
        result = out.convert(img.mode)
        
        im = lb.convert("RGBA")
        rot2 = im.rotate(ranges['rotate'][3]*a)
        ff = Image.new("RGBA", rot2.size, (5,) * 4)
        out2 = Image.composite(rot2, ff, rot2)
        result2 = out2.convert(lb.mode)
               
    elif n == 7:
        result = img.transform(
        img.size, Image.AFFINE, (1, 0, 0, ranges['shearY'][3]* a, 1, 0),
        Image.BICUBIC, fillcolor=(0))
        
        result2 = lb.transform(
        img.size, Image.AFFINE, (1, 0, 0, ranges['shearY'][3]* a, 1, 0),
        Image.BICUBIC, fillcolor=5)
    '''
    if n == 3 or  n == 2:
        enhancer = ImageEnhance.Brightness(img)
        result = enhancer.enhance( 1+ranges['brightness'][5] * a)
        result2 = lb
        
    elif n == 4 or  n == 1: 
        enhancer = ImageEnhance.Contrast(img)
        result = enhancer.enhance( 1+ranges['contrast'][5] * a)
        result2 = lb
    elif n == 5 or n == 0:
        result = np.fliplr(np.asarray(img))
        result2 = np.fliplr(np.asarray(lb))
        
    return np.asarray(result), np.asarray(result2)

        

#%%
data_rootpath = os.getcwd() 
file_dirs = []
for file in os.listdir(data_rootpath + os.sep + 'roi'):
    if file.endswith(".zip"):
        file_dirs.append(file)
labels = np.ones((265, 3100, 2, len(file_dirs)*2))*5
images = np.zeros((265, 3100, 1, len(file_dirs)*2))
l =  len(file_dirs)*2
i = 0
for dirs in file_dirs:
    regex = re.compile(r'\d+')
    index = regex.findall(dirs)[0]
    img = plt.imread(data_rootpath + os.sep + 'images' + os.sep + f'image_{index}.jpg' )
    images[:,:,0,i] = img[:,:,0]
    labels[:,:,0,i] = get_label(data_rootpath + os.sep + 'roi' + os.sep + dirs)
    result_img , result_lb = data_aug(img, labels[:,:,0,i], i)
    images[:,:,0,l-1-i] = result_img[:,:,0]
    labels[:,:,0,l-1-i] = result_lb
    i += 1
w = weight(labels[:,:,0,:],3,6)    
labels[:,:,1,:] = w

Set = np.ones((1,len(labels[0,0,0,:])))
Set[0,int(len(Set[0,:])*0.8):] = 3 
with h5py.File(data_rootpath + '/set.h5', 'w') as f:
    f.create_dataset('Set', data = Set) 
with h5py.File(data_rootpath+'/Data.h5', 'w') as f:
    f.create_dataset('oct_dataset', data = images)  
with h5py.File(data_rootpath+'/label.h5', 'w') as f:
    f.create_dataset('oct_labels', data = labels)  

#%%
def get_label(directory):
    # get roi info from a RoiSet.zip 
    roi = read_roi_zip(directory)      
    # a matrix for labels
    label = np.zeros((5,3100))
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
    lb = create_labels(label)
    return lb
        

def label_a_col(label_for_col):
    """
    takes in a 1d list indicating the layer transition of a column and recreate the actual image label
    with size of the image
    the endpoints inclusive or exclusive?
    """
    # SIZE: (496,523)
    col = np.ones(265) * 5 # TODO: 10 classes
    for i in range(len(label_for_col)): # -4 creates 5 classes
        #index = 4-i # label row index 0-7
        #index = 3-i # 5c
        #col[:int(label_for_col[index])] = index + 1 # class number 1-8
        if label_for_col[i] == 0:
            continue
        col[int(label_for_col[i]):] = i + 1 # class number 1-8
    return col
    

    
def create_labels(labels):
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


def weight(labels, w1, w2):
    """
    w(x) = 1 + w1 * I(|gradient(l(x))|) + w2 * I(l(x) = L)
    L = retinal layers and fluid masses
    :param labels: a list of labels
    :param w1:
    :param w2:
    :return:
    """    
    weights = np.zeros((labels.shape)) #(496,523,11)
    labels = np.array(labels)
    for i in range(labels.shape[2]):
        label = labels[:,:,i]
        I_w2 = label
        label = np.gradient(label)
        grad = list(map(abs, label))
        abs_grad = grad[0]+grad[1]
        abs_grad[abs_grad>0] = 1
        I_w1 = abs_grad
        pos1 = I_w2 == 5 # label of background
        pos2 = I_w2 == 6 #TODO:  background 
        I_w2 = 1*(~np.array(pos1 + pos2))
        weight = 1+ w1*I_w1 + w2*I_w2
        weights[:,:,i] = weight
        ####
        #plt.imshow(weights[:,:,i])
        #plt.show()
        #####
    return weights


 

