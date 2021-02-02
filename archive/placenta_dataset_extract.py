#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:46:53 2020

@author: skalalab
"""

# ECG: I believe this originally for Kayvans only segmented datset using Weka 

import numpy as np
import matplotlib.pyplot as plt
import cv2

def image_show(image, nrows=1, ncols=1):  # , cmap='gray'

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    ax.imshow(image)  # , cmap ='gray'
    ax.axis('off')
    return fig, ax
#%%
    


# %%
if __name__ == '__main__':
 
   # imgpath = 'SUMPOS2/SUM_POS2-1.tif'
    labelpath = '/Users/maywu/Desktop/Skala_lab/placenta_segmentation/labels/labels_0.jpg'
    imgpath = '/Users/maywu/Desktop/Skala_lab/placenta_segmentation/images/image_0.jpg'
    label = plt.imread(labelpath)
    image_show(label[:,1000:1500,:])
    
    img = plt.imread(imgpath)
    image_show(img[:,1000:1500,:])
    
    
    ## convert to hsv
    hsv = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)
    
    ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask_green = cv2.inRange(hsv, (51, 40, 20), (90, 255,255))
    mask_pink = cv2.inRange(hsv, (130, 25, 20), (179, 255,255))
    mask_blue_light = cv2.inRange(hsv, (11, 20, 20), (50, 255,255))
    mask_blue_dark = cv2.inRange(hsv, (0, 0, 0), (10, 255,255))
    ## slice the green
    imask = mask_green>0 
    imask2 = mask_pink>0
    imask3 = mask_blue_light>0
    imask4 = mask_blue_dark >0
    
    
    green = np.zeros_like(label, np.uint8)
    
    pink = np.zeros_like(label, np.uint8)
    blue_light = np.zeros_like(label, np.uint8)
    blue_dark = np.zeros_like(label, np.uint8)
    
    green[imask] = label[imask]
    pink[imask2] = label[imask2]
    blue_light[imask3] = label[imask3]
    blue_dark[imask4] = label[imask4]
    
    green[green>0] = 3
    blue_light[blue_light>0] = 1
    pink[pink>0] = 2
    blue_dark[blue_dark>0] = 4
    
    green =  blue_light + pink + green + blue_dark
    green = green + 1
    image_show(img[:,800:1800,0])
    print(np.max(green[:,100:3000,0]))
