# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:30:13 2020

@author: econtrerasguzman
"""

#!/usr/bin/env python3

# from scipy.io import loadmat
import matplotlib.pylab as plt
import numpy as np
from read_roi import read_roi_zip
from skimage.draw import line
import skimage 
import skimage.transform

# import h5py
# from pathlib import Path
# import random
# import sys
# import tempfile
# import shutil
# import relaynet_utils




# https://www.osapublishing.org/DirectPDFAccess/752C0312-E5B1-4ED9-888AAD072B65F4C7_312754/boe-6-4-1172.pdf?da=1&id=312754&seq=0&mobile=no
# from paper: 
#     An ophthalmologist then identified six
# patients imaged in clinic using the standard Spectralis (Heidelberg Engineering, Heidelberg,
# Germany) 61-line volume scan protocol with severe DME pathology and varying image
# quality. Averaging of the B-scans was determined by the photographer, and ranged from 9 to
# 21 raw images per averaged B-scan. The volumetric scans were Q = 61 B-scans × N = 768


# To generate the target classes for classifier training, we manually segmented fluid-filled
# regions and semi-automatically segmented all eight retinal layer boundaries following the
# definitions in Fig. 3. This was done for 12 B-scans within the training data set (two from each
# volume). The B-scans selected consisted of six images near the fovea (B-scan 31 for all(i)
# volumes) and six peripheral images (B-scans 1, 6, 11, 16, 21, and 26, one for each of the six
# volumes). We then used the manual segmentations to assign the true class for each pixel, with
# a total of eight possible classes defined in Table 1 and the classified result shown in Fig. 4(a).


##NOTE THEY ARE TRAINING THEIR IMAGES IN 512X512 SIZES --> SEE KERNEL DIMENSIONS

# SEG_LABELS_LIST = [
#     {"id": -1, "name": "void", "rgb_values": [0, 0, 0]},
#     {"id": 0, "name": "Region above the retina (RaR)", "rgb_values": [128, 0, 0]},
#     {"id": 1, "name": "ILM: Inner limiting membrane", "rgb_values": [0, 128, 0]},
#     {"id": 2, "name": "NFL-IPL: Nerve fiber ending to Inner plexiform layer", "rgb_values": [128, 128, 0]},
#     {"id": 3, "name": "INL: Inner Nuclear layer", "rgb_values": [0, 0, 128]},
#     {"id": 4, "name": "OPL: Outer plexiform layer", "rgb_values": [128, 0, 128]},
#     {"id": 5, "name": "ONL-ISM: Outer Nuclear layer to Inner segment myeloid", "rgb_values": [0, 128, 128]},
#     {"id": 6, "name": "ISE: Inner segment ellipsoid", "rgb_values": [128, 128, 128]},
#     {"id": 7, "name": "OS-RPE: Outer segment to Retinal pigment epithelium", "rgb_values": [64, 0, 0]},
#     {"id": 8, "name": "Region below RPE (RbR)", "rgb_values": [192, 0, 0]}];

# ECG see below, dimension order rin these matrices are wrong
# imdb.images.data is a 4D matrix of size: [height, width, channel, NumberOfData]
# imdb.images.labels is a 4D matrix of size: [height, width, 2, NumberOfData] ---> 1st Channel is class (1,2,... etc), 2nd channel is Instance Weights (All voxels with a class label is assigned a weight, details in paper)
# imdb.images.set is [1,NumberOfData] vector with entries 1 or 3 indicating which data is for training and validation respectively.



"""
This script generates the training dataset for ReLayNet from the Duke dataset. 
change the directory names when use. If you want to reduce the number of classes
to be classified, modify the places maked by TODOs. you can also change the 
w1 and w2 values for weight generation.
"""


#### Taken from relaynet original repo
SEG_LABELS_LIST = [
    {"id": -1, "name": "void", "rgb_values": [0, 0, 0]},
    {"id": 0, "name": "Region above the retina (RaR)", "rgb_values": [128, 0, 0]},
    {"id": 1, "name": "ILM: Inner limiting membrane", "rgb_values": [0, 128, 0]},
    {"id": 2, "name": "NFL-IPL: Nerve fiber ending to Inner plexiform layer", "rgb_values": [128, 128, 0]},
    {"id": 3, "name": "INL: Inner Nuclear layer", "rgb_values": [0, 0, 128]},
    {"id": 4, "name": "OPL: Outer plexiform layer", "rgb_values": [128, 0, 128]},
    {"id": 5, "name": "ONL-ISM: Outer Nuclear layer to Inner segment myeloid", "rgb_values": [0, 128, 128]},
    {"id": 6, "name": "ISE: Inner segment ellipsoid", "rgb_values": [128, 128, 128]},
    {"id": 7, "name": "OS-RPE: Outer segment to Retinal pigment epithelium", "rgb_values": [64, 0, 0]},
    {"id": 8, "name": "Region below RPE (RbR)", "rgb_values": [192, 0, 0]}];
    #{"id": 9, "name": "Fluid region", "rgb_values": [64, 128, 0]}];
    
def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)

######################################## Duke Dataset functions below ###############################################
def get_annotated_img_index(manual_layers):
    """
    find the index of images which have been annotated with labels and return 
    the index, the range of pixel being labeled. 
    return:
        @valid_img_index: index of annotated images as 1d list 
        @valid_range: list of 2 elements
    """
    valid_img_index = []
    valid_range = []
    ####
    _, _, num_images = manual_layers.shape
    
    for num_frame in np.arange(num_images):
        mask = manual_layers[...,num_frame]
        
        if np.nansum(mask) != 0: # get only layers with data
            valid_img_index.append(num_frame)
            
            #show mask
            # plt.imshow(mask)
            # plt.show()
           
            #calculate min and max values
            one_dimension = np.sum(mask, axis=0)
            list_not_nan= np.where(~np.isnan(one_dimension))[0] # find values
            valid_min = list_not_nan[0]
            valid_max = list_not_nan[-1]
            print(f"layer: {num_frame}, valid [max,min]: [{valid_min},{valid_max}]")
            
            # get valid min max values --> could probably remove
            if valid_min not in valid_range: 
                valid_range.append(valid_min)
            if valid_max not in valid_range:
                valid_range.append(valid_max)
    valid_range.sort()
    
    return valid_img_index, valid_range

def generate_weights(mask_layers, w1, w2, show_weights_mask = False):
    """
    This takes in a mask already created
    
    w(x) = 1 + w1 * I(|gradient(l(x))|) + w2 * I(l(x) = L)
    L = retinal layers and fluid masses
    
    :param mask_layers: a single mask of labels for a single image
    :param w1:
    :param w2:
    :return:
    """    
    
    # in spyder change figures to show higher, otherwise it shows gaps in layers
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 300

    # initialize mask to ones so bg, top and bottom layers are ones
    img_rows, im_cols = mask_layers.shape 
    mask_weights = np.ones((img_rows, im_cols)) #(496,523,11)
    
    # exclude first two layers (void, top of retina) and the last below retina
    # TODO find better way to define these layers
    #exclude void, area above and below retina [-1,0,last_layer] respectively
    valid_layers = np.unique(mask_layers)[2:-1]
    
    eq_coeff = 1
    # iterate through columns in mask
    for idx, col in enumerate(mask_layers.transpose()):

        # label single column in mask
        labeled_col = np.ones((img_rows)) # first term in equation is 1
        for pos, pixel in enumerate(col):
            
            # add weight 1 if a layer
            if pixel in valid_layers:
                labeled_col[pos] += w1
            
            # add weight 2 if transition pixel
            if pos == 0: # skip first pixel, no previous pixel
                pass
            # fill middle pixels
            elif pixel != col[pos - 1]: # look at previous pixel
                if pixel in valid_layers: 
                    labeled_col[pos] += w2 # add weight to current pixel
                else: # it's region below RPE or border of void
                    labeled_col[pos - 1] = eq_coeff + w1 + w2 # transition pixel
                    
        mask_weights[:,idx] = labeled_col
    
    if show_weights_mask:
        plt.imshow(mask_weights, interpolation='nearest')
        plt.show()
    return mask_weights


def generate_labels_mask(transition_layers, img_rows, img_cols, show_mask = False):
    ## generates a mask given 
    
    ## Remove nan's in label mask by filling void pixels with previous pixel value on same layer
    
    #calculate min and max range of values
    one_dimension = np.sum(transition_layers, axis=0)
    list_not_nan= np.where(~np.isnan(one_dimension))[0] # find values where array is not NaN
    valid_min = list_not_nan[0]
    valid_max = list_not_nan[-1]
    
    #fill in nan's in transition_layers within range
    for idx_row, row in enumerate(transition_layers): # iterate through rows
        for idx_col, pixel in enumerate(row): # iterate through pixel in row
            if idx_col == 0: # skip first pixel
                continue
            if idx_col > valid_min and idx_col <=valid_max: # label in range
                if np.isnan(pixel):
                    # previous non_nan value 
                    transition_layers[idx_row, idx_col] = row[idx_col-1]
                
    
    #transition_layers = np.nan_to_num(transition_layers).astype(int) # replace nans with zeros
    rows, cols = transition_layers.shape
    mask = np.zeros((img_rows, img_cols))
    
    # iterate through each col of transition layers to create mask
    for col in np.arange(cols): 
        mask[:,col] = _label_column(transition_layers[:,col], img_rows)
    
    if show_mask:
        plt.imshow(mask)
        plt.show()
        
    return mask

def _label_column(transition_layers_pixel, img_rows):
    """
    this function finds the start and stop pixels for a layer and labels it accordingly
    given the number of transition pixels and layers it can identify (8 pixels can
    separte 7 layers)

    Parameters
    ----------
    transition_layers_pixel : array
        1d array of transition pixel for a single column
    img_rows : int
        rows in label column for initializing array
    fluid_layer: bool
        layers are labeled the same

    Returns
    -------
    mask_col : TYPE
        returns a labled column.

    """
    
    # skip empty columns
    if np.count_nonzero(np.isnan(transition_layers_pixel)) == transition_layers_pixel.size:
        return
    
    # create column mask to store column labels
    mask_col = np.empty((img_rows))
    mask_col[:] = np.NaN
    
    
    # include last pixel in row(plus one because like that applies labels is non inclusive
    # on last pixel)
    transition_layers_pixel = np.append(transition_layers_pixel, (img_rows + 1)) 
    
    # generate list of labels, a list of 8 transition pixels can label 9 layers 
    list_layers = np.arange(transition_layers_pixel.size) 
    
    # initialize where to start labeling
    #idx_start = np.NaN # to label only retina layers
    idx_start = 0 # for starting to label at the bottom of image (Region above the retina (RaR))

    for pos, (transition_pixel, label) in enumerate(zip(transition_layers_pixel, list_layers)):
        
        # at the beginning, initialize idx_start to first entry
        # in transition pixels
        # this also helps skip sections where nan is found
        if np.isnan(idx_start):
            idx_start = transition_pixel
            continue
        
        # get next "idx_stop" pixel to know when to stop labeling layer
        #else :#pos < len(transition_layers_pixel):
        idx_stop = transition_pixel
        # this condition treats sections with NaN within layers
        if np.isnan(idx_stop):
            idx_start = np.NaN # next layer is missing
            continue
        
        #finally label layer in column
        mask_col[int(idx_start):int(idx_stop)] = label  # non inclusive index, like python 0-3 labels 0-2
        # next start pixel is previous stop pixel
        idx_start = idx_stop
    
    
    return mask_col


######################################## OCT dataset Modifications below ###############################################


def generate_labels_mask_oct(transition_layers, img_rows, img_cols, show_mask = False):
    ## generates a mask given 
    
    ## Remove nan's in label mask by filling void pixels with previous pixel value on same layer
    
    #calculate min and max range of values
    one_dimension = np.sum(transition_layers, axis=0)
    list_not_nan= np.where(~np.isnan(one_dimension))[0] # find values where array is not NaN
    valid_min = list_not_nan[0]
    valid_max = list_not_nan[-1]
    
    #fill in nan's in transition_layers within range
    for idx_row, row in enumerate(transition_layers): # iterate through rows
        for idx_col, pixel in enumerate(row): # iterate through pixel in row
            if idx_col == 0: # skip first pixel
                continue
            if idx_col > valid_min and idx_col <=valid_max: # label in range
                if np.isnan(pixel):
                    # previous non_nan value 
                    transition_layers[idx_row, idx_col] = row[idx_col-1]
                
    
    #transition_layers = np.nan_to_num(transition_layers).astype(int) # replace nans with zeros
    rows, cols = transition_layers.shape
    mask = np.zeros((img_rows, img_cols))
    
    # iterate through each col of transition layers to create mask
    for col in np.arange(cols): 
        mask[:,col] = _label_column_oct(transition_layers[:,col], img_rows)
    
    if show_mask:
        plt.imshow(mask)
        plt.show()
        
    return mask




def _label_column_oct(transition_layers_pixel, img_rows):
    """
    this function finds the start and stop pixels for a layer and labels it accordingly
    given the number of transition pixels and layers it can identify (8 pixels can
    separte 7 layers)

    Parameters
    ----------
    transition_layers_pixel : array
        1d array of transition pixel for a single column
    img_rows : int
        rows in label column for initializing array
    fluid_layer: bool
        layers are labeled the same

    Returns
    -------
    mask_col : TYPE
        returns a labled column.

    """
    
    # skip empty columns
    if np.count_nonzero(np.isnan(transition_layers_pixel)) == transition_layers_pixel.size:
        return
    
    # create column mask to store column labels
    mask_col = np.empty((img_rows))
    mask_col[:] = np.NaN
    
    
    # include last pixel in row(plus one because like that applies labels is non inclusive
    # on last pixel)
    transition_layers_pixel = np.append(transition_layers_pixel, (img_rows + 1)) 
    
    # generate list of labels, a list of 8 transition pixels can label 7 layers 
    list_layers = np.arange(transition_layers_pixel.size) 
    
    # initialize where to start labeling
    #idx_start = np.NaN # to label only retina layers
    idx_start = 0 # for starting to label at the bottom of image (Region above the retina (RaR))

    for pos, (transition_pixel, label) in enumerate(zip(transition_layers_pixel, list_layers)):
        
        # at the beginning, initialize idx_start to first entry
        # in transition pixels
        # this also helps skip sections where nan is found
        if np.isnan(idx_start):
            idx_start = transition_pixel
            continue
        
        # get next "idx_stop" pixel to know when to stop labeling layer
        #else :#pos < len(transition_layers_pixel):
        idx_stop = transition_pixel
        # this condition treats sections with NaN within layers
        if np.isnan(idx_stop):
            # idx_start = np.NaN # next layer is missing
            # continue
            idx_stop = img_rows
        
        #finally label layer in column
        mask_col[int(idx_start):int(idx_stop)] = label  # non inclusive index, like python 0-3 labels 0-2
        
        if idx_stop == img_rows: # filled all rows
            break
        # next start pixel is previous stop pixel
        idx_start = idx_stop
    
    
    return mask_col


def generate_weights_oct(mask_layers, w1, w2, show_weights_mask = False):
    """
    This takes in a mask already created
    
    w(x) = 1 + w1 * I(|gradient(l(x))|) + w2 * I(l(x) = L)
    L = retinal layers and fluid masses
    
    :param mask_layers: a single mask of labels for a single image
    :param w1:
    :param w2:
    :return:
    """    

    # initialize mask to ones so bg, top and bottom layers are ones
    img_rows, im_cols = mask_layers.shape 
    mask_weights = np.ones((img_rows, im_cols)) #(496,523,11)
    
    # exclude first two layers (void, top of retina) and the last below retina
    # TODO find better way to define these layers
    #exclude void, area above and below retina [-1,0,last_layer] respectively
    # valid layers are layers that are part of the tissue, area above and below are background
    valid_layers = mask_layers[~np.isnan(mask_layers)]
    valid_layers = np.unique(valid_layers)[1:-1]
    
    eq_coeff = 1
    # iterate through columns in mask
    for idx, col in enumerate(mask_layers.transpose()):

        # label single column in mask
        labeled_col = np.ones((img_rows)) # first term in equation is 1
        for pos, pixel in enumerate(col):
            
            # add weight 1 if a layer
            if pixel in valid_layers:
                labeled_col[pos] += w1
            
            # add weight 2 if transition pixel
            if pos == 0: # skip first pixel, no previous pixel
                continue
            # fill middle pixels
            elif pixel != col[pos - 1]: # look at previous pixel
                if pixel in valid_layers: 
                    labeled_col[pos] += w2 # add weight to current pixel
                else: # it's region below RPE or border of void
                    labeled_col[pos - 1] = eq_coeff + w1 + w2 # transition pixel
                    
        mask_weights[:,idx] = labeled_col
    
    if show_weights_mask:
        plt.imshow(mask_weights, interpolation='nearest')
        plt.show()
    return mask_weights

def roi_to_labels_mask(path_roi_file, im_rows, im_cols, show_image = False):
    
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
    labels : numpy array
        the mask created from the roi file. lb is an image.

    """
    # get roi info from a RoiSet.zip 
    rois = read_roi_zip(path_roi_file)   
    
    n_layers = len(rois.keys())# with n lines you can label n+1 layers
    # a matrix for labels
    labels = np.empty((n_layers, im_cols)) 
    labels[:] = np.NaN
    
    # iterate through layers and make array
    # should create n lists with each list containing pixels for one layer
    for layer_idx, key in enumerate(rois.keys()):
        ## potentially cast to int here
        vertices_col_vals = rois[key]['x'] # columns
        vertices_row_vals = rois[key]['y'] # rows 
        
        ### round subpixel masks to int 
        vertices_col_vals = [int(round(val)) for val in vertices_col_vals]
        vertices_row_vals = [int(round(val)) for val in vertices_row_vals]
        
        # create line by iterating through the vertices
        for list_pixel_idx, (curr_pixel_row, curr_pixel_col) in enumerate(zip(vertices_row_vals,vertices_col_vals)):
            if (list_pixel_idx + 1) == len(vertices_col_vals): # this is the last pixel,return
                break
            # generate line
            next_row_pixel = vertices_row_vals[list_pixel_idx +1]
            next_col_pixel = vertices_col_vals[list_pixel_idx +1]
            layer_pixels = line(curr_pixel_row, curr_pixel_col,next_row_pixel ,next_col_pixel)
            
            # fill in labels array
            for pos, (pixel_row, pixel_col) in enumerate(zip(*layer_pixels)):
                labels[layer_idx, pixel_col] =  pixel_row # save only row value
    if show_image:
        plt.imshow(labels)
        plt.show()
    
    return labels
#%% augmentations
        
# mirror x is good
# translations are good (up to edges of frame). 
# scale in y up to +/-10%. 
# rotate up to +/-10 degrees. 
# always crop to # the original dimension. zero pad if needed. I think!

# image = list_images[0].transpose()
# mask = list_inferences[0]
# plt.imshow(image)
# plt.imshow(mask)
# # mirror 


# make a decorator to show image

# default to mirror vertically (OCT images are vertical)
def mirror_array(array, axis = 1, show_image = True):
    im_mirrored = np.flip(array,axis)
    
    if show_image:
        plt.imshow(im_mirrored)
        plt.show()
    return im_mirrored

def translations(image, labels, weights, show_image = False):
    
    # calculate amount to translate by using the mask
    
    #remove bottom layer
    mask_layers_only = labels.copy()
    mask_layers_only[labels == np.unique(labels)[0]] = 0 # remove top bg layer
    mask_layers_only[labels == np.unique(labels)[-1]] = 0 # remove bottom bg layer
    mask_flat = np.sum(mask_layers_only, axis=0) # flatten image into 1d array
    mask_valid_data_indices = np.argwhere(mask_flat > 0) # calculate indices where data > 0 (find borders)
    
    # calculate left translate value
    transpose_left_value = np.min(mask_valid_data_indices) # find bottom shift value
    
    # calculate right translate value
    transpose_right_idx = np.max(mask_valid_data_indices) # find top shift value
    mask_length = mask_flat.shape[0]
    transpose_right_value = mask_length - transpose_right_idx
    
    # translate mask and and image right
    translate_image_right = np.roll(image, transpose_right_value, axis=1) # axis=1 is cols
    translate_labels_mask_right = np.roll(labels, transpose_right_value,axis=1)
    translate_weights_mask_right = np.roll(weights, transpose_right_value, axis=1)
    
    if show_image:
        plt.imshow(translate_image_right)
        plt.imshow(translate_labels_mask_right)
        plt.imshow(translate_weights_mask_right)
    
    #translate bottom
    translate_image_left = np.roll(image, -transpose_left_value,axis=1)
    translate_labels_mask_left = np.roll(labels, -transpose_left_value,axis=1)
    translate_weights_mask_left = np.roll(weights, -transpose_left_value,axis=1)
    if show_image:
        plt.imshow(translate_image_left)
        plt.imshow(translate_labels_mask_left)
        plt.imshow(translate_weights_mask_left)

    return (translate_image_left, translate_image_right), (translate_labels_mask_left, translate_labels_mask_right), (translate_weights_mask_left, translate_weights_mask_right)

def rotate_image(image, degrees, show_image = False):
    rotated = skimage.transform.rotate(image, angle=degrees, preserve_range=True, order=0) # order=0 is nearest neighbor
    if show_image:
        plt.imshow(rotated)
        plt.show()
    return rotated

def rotate_mask(mask, degrees, fill_constant_value, show_image = False):
    rotated = skimage.transform.rotate(mask,angle=degrees, preserve_range=True, order=0, mode='constant', cval=fill_constant_value)
    #rotated = np.ceil(rotated).astype(int) # convert floats to integers
    if show_image:
        plt.imshow(rotated)
        plt.show()
    return rotated

# def rotate_weights_mask(mask, degrees, show_image = False):
#     rotated = skimage.transform.rotate(mask,angle=degrees, preserve_range=True, order=0)
#     #rotated = np.ceil(rotated).astype(int) # convert floats to integers
#     if show_image:
#         plt.imshow(rotated)
#         plt.show()
#     return rotated

