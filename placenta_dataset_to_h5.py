# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:44:14 2020

@author: econtrerasguzman
"""

from pathlib import Path
import relaynet_utils as ru
import tifffile
# from bresenham import bresenham
# from skimage.draw import line
# from read_roi import read_roi_zip
import numpy as np
import matplotlib.pylab as plt
import random
import h5py
import skimage.transform
from skimage.draw import line
from read_roi import read_roi_zip

# in spyder change figures to show higher, otherwise it shows gaps in layers
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400

# path_segmented_dataset = Path("Z:/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/0-segmentation_completed")
#path_segmented_dataset = Path("C:/Users/econtrerasguzman/Desktop/0-segmentation_completed")
path_segmented_dataset = Path("/run/user/1000/gvfs/smb-share:server=skala-dv1.discovery.wisc.edu,share=ws/skala/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/0-segmentation_completed")
print(f"path_segmented_datset exists: {path_segmented_dataset.exists()}")

path_h5_output = path_segmented_dataset.parent / "0-h5"

# path_output = Path("Z:\0-Projects and Experiments\KS - OCT membranes\oct_dataset_3100x256\0-segmented_dataset")

list_images = []
list_labels = []
list_weights = []
not_found = 0


# path_roi = Path("/home/skalalab/Desktop/test_roi_set.zip")
# rois = read_roi_zip(path_roi) 

# iterate through each data folder
for img_folder in list(path_segmented_dataset.glob("*_amniochorion_*")):
# for img_folder in [list(path_segmented_dataset.glob("*_amniochorion_*"))[-1]]:
    
    
    print(f"***** Processing Directory: {img_folder.name}")
    path_images = img_folder / "images"
    path_rois = img_folder / "roi_files"
    
    # iterate through each roi
    for path_roi_file in list(path_rois.glob("*.zip")):
        print(f"roi found: {path_roi_file.name}")

        # image path
        path_image = path_images / f"{path_roi_file.stem}.tiff"
        
        # validate paths
        # suspect too long image paths raise FileNotFound exception
        try:
            if path_roi_file.exists() == False :
                print("roi file not found")
                raise FileNotFoundError
            
            if path_image.exists() == False :
                # print(f"image file not found")
                raise FileNotFoundError
        except:
            not_found += 1
            print(f"file not found: {str(path_roi_file.name)}")
            continue
        
        # Get image file 
        image = tifffile.imread(str(path_image))
        
        # some images are three channel, grab first channel
        if len(image.shape) == 3:
            image = image[...,0] # (rows,cols, channe )grab first channel
        
        list_images.append(image)
        im_rows, im_cols = image.shape
        
        # get labels mask    
        mask_labels_encoded = ru.roi_to_labels_mask(path_roi_file,im_rows, im_cols )
        
        mask_labels = ru.generate_labels_mask_oct(mask_labels_encoded, im_rows, im_cols)
        
        # side NaN's in mask are considered background, label them as 0
        mask_labels[np.isnan(mask_labels)] = 0
        
        # offset labels so they start at 1
        mask_labels += 1
        list_labels.append(mask_labels)
        
        # plt.imshow(mask_labels)
        # plt.show()
        
        # get weights mask
        w1 = 10
        w2 = 50
        mask_weights = ru.generate_weights_oct(mask_labels, w1, w2)
        list_weights.append(mask_weights)
        # plt.imshow(mask_weights)
        # plt.show()
        

print(f"number of images: {len(list_images)}")
print(f"number of labels masks: {len(list_labels)}")
print(f"number of weight masks: {len(list_weights)}")
print(f"images not found: {not_found}")

#%% Apply augmentations
list_images_aug = []
list_labels_aug = []
list_weights_aug = []


for pos, (image, mask_labels, mask_weights) in enumerate(zip(list_images, list_labels, list_weights)):
    pass
    print(f"Augmenting image: {pos+1}/{len(list_images)}")
          
    ##### MIRROR DATA
    debug = False
    im_mirrored = ru.mirror_array(image, show_image=debug)
    labels_mirrored = ru.mirror_array(mask_labels, show_image=debug)
    weights_mirrored = ru.mirror_array(mask_weights, show_image=debug)
    
    list_images_aug.append(im_mirrored) 
    list_labels_aug.append(labels_mirrored)
    list_weights_aug.append(weights_mirrored)

    ##### TRANSLATE DATA
    # translate returns tuples with left and right shifted images
    translated_images, translated_labels, translated_weights = ru.translations(image, mask_labels, mask_weights, show_image=debug)
    for t_im, t_labels, t_weights in zip(translated_images, translated_labels, translated_weights):
        list_images_aug.append(t_im)
        list_labels_aug.append(t_labels)
        list_weights_aug.append(t_weights)
   
    ##### SCALE DATA
    num_rows, num_cols = image.shape
    resize_percent = 1.1
    scaled_rows_size = int(num_rows*resize_percent)
    scaled_cols_size = int(num_cols*resize_percent)
    
    
    im_scaled = skimage.transform.resize(image,(scaled_rows_size, scaled_cols_size), preserve_range = True, anti_aliasing=False, order=0)
    labels_scaled = skimage.transform.resize(mask_labels,(scaled_rows_size, scaled_cols_size), preserve_range = True, anti_aliasing=False, order=0)
    weights_scaled = skimage.transform.resize(mask_weights,(scaled_rows_size, scaled_cols_size), preserve_range = True, anti_aliasing=False, order=0)
    
    # clean up mask
    # weight_bg, weight_layer, weight_transition = np.unique(mask_weights)
    # weights_scaled[(weights_scaled > weight_layer) * (weights_scaled < (weight_transition*0.6))] = weight_layer # arbitrary threshold based on looking at the mask
    # weights_scaled[weights_scaled >= weight_transition*0.6] = weight_transition
    # weights_scaled[weights_scaled < weight_layer] = weight_bg
    
    # resize to original rows/cols
    # calculate center of image
    rows_offset = scaled_rows_size - num_rows
    cols_offset = scaled_cols_size - num_cols
    
    # calculate slicing for proper dimensions
    rows_start = int(np.floor(rows_offset/2))
    rows_end = int(np.ceil(rows_offset/2))
    cols_start = int(np.floor(cols_offset/2))
    cols_end = int(np.ceil(cols_offset/2))
    
    # add to dataset and crop to dimensions
    list_images_aug.append(im_scaled[rows_start:-rows_end,cols_start:-cols_end])
    list_labels_aug.append(labels_scaled[rows_start:-rows_end,cols_start:-cols_end])
    list_weights_aug.append(weights_scaled[rows_start:-rows_end,cols_start:-cols_end])
    
    
    ##### REFOCUSING AUGMENTATION
    # collapse into 2d to find if placeta near the top
    projection_rows = np.sum(mask_labels-1, axis=1) # offset labels so top bg gets value of zero, then sum across cols
    percent_threshold = 0.1 # threshold reached for 50% crop and reposition
    threshold_pixels = percent_threshold * len(projection_rows)
    if np.sum(projection_rows[:int(np.floor(threshold_pixels))]) > 20: # if image near the top of the focal field, augment | arbitrary 20 to avoid augmentation from noise
        
        print(f"Refocusing augmenting for image {pos+1}")    
        # roll down and fill top with zeros    
        im_rows, _ = image.shape
        roll_value = int(np.floor(im_rows/2)) # roll by 50%
        
        # for array in [image, mask_labels, mask_weights]:
        #     plt.imshow(array)
        #     plt.show()
        
        # roll down
        rolled_image = np.roll(image, roll_value, axis=0)
        rolled_labels = np.roll(mask_labels, roll_value, axis=0)
        rolled_weights = np.roll(mask_weights, roll_value, axis=0)
        
        # blank top portion
        rolled_image[:roll_value] = 0 # make bg 
        rolled_labels[:roll_value] = np.unique(mask_labels)[0] # first layer is bg
        rolled_weights[:roll_value] = np.min(mask_weights) # this is bg weight
        
        list_images_aug.append(rolled_image)
        list_labels_aug.append(rolled_labels)
        list_weights_aug.append(rolled_weights)
    
        # for array in [rolled_image, rolled_labels, rolled_weights]:
        #     plt.imshow(array)
        #     plt.show()
        
        # plt.imshow(np.random.random((1,1024)))
        # plt.show()
    
    fill_constant_value_labels = np.unique(mask_labels)[0] # fill with bg , label 0
    fill_constant_value_weights = np.unique(mask_weights)[0]
    # rotate 
    im_rotate_plus_10 = ru.rotate_image(image, 10, show_image=debug)
    labels_rotate_plus_10 = ru.rotate_mask(mask_labels, 10,fill_constant_value_labels, show_image=debug)  
    weights_rotate_plus_10 = ru.rotate_mask(mask_weights, 10, fill_constant_value_weights, show_image=debug)
    
    list_images_aug.append(im_rotate_plus_10)
    list_labels_aug.append(labels_rotate_plus_10)
    list_weights_aug.append(weights_rotate_plus_10)
    
    im_rotate_minus_10 = ru.rotate_image(image, -10, show_image=debug)
    labels_rotate_minus_10 = ru.rotate_mask(mask_labels, -10, fill_constant_value_labels, show_image=debug)  
    weights_rotate_minus_10 = ru.rotate_mask(mask_weights, -10, fill_constant_value_weights, show_image=debug)
    
    list_images_aug.append(im_rotate_minus_10)
    list_labels_aug.append(labels_rotate_minus_10)
    list_weights_aug.append(weights_rotate_minus_10)

#%% add lists to dataset 
    
list_images += list_images_aug
list_labels += list_labels_aug
list_weights += list_weights_aug

#%%
num_images = len(list_images)
             
# paper stated these dimensions, pad to this below
paper_arr_rows, paper_arr_cols = (3100,512)
            
# imdb.images.data is a 4D matrix of size: [height, width, color channel (1ch), NumberOfData]
image_color_channels = 1
h5_data = np.zeros((num_images, image_color_channels, paper_arr_rows, paper_arr_cols))

# imdb.images.labels is a 4D matrix of size: [height, width, 2, NumberOfData] 
dim_class_and_weights = 2 # ---> 1st Channel is class (1,2,... etc), 2nd channel is Instance Weights (All voxels with a class label is assigned a weight, details in paper)
h5_labels = np.zeros((num_images, dim_class_and_weights, paper_arr_rows, paper_arr_cols))

# imdb.images.set is [1,NumberOfData] vector with entries 1 or 3 indicating which data is for training and validation respectively.
# h5_set = np.zeros((1, num_images))
# ID for training or testing
dim_train_test_id = 1
h5_set = np.ones((num_images, dim_train_test_id))

    
# shuffle lists
# RANDOMIZE!
list_all = list(zip(list_images, list_labels, list_weights))
random.shuffle(list_all)    

# iterate through the number of images and make h5 stacks
for idx, (image, labels, weights) in enumerate(list_all):
    pass
    print(f"Adding image to h5 stack {idx+1}/{len(list_images)}")
    ###
    #they orient their images vertically, transpose
    image = image.transpose()
    labels = labels.transpose()
    weights = weights.transpose()
    
    # get image shape
    img_rows, img_cols = image.shape
    
    # calculate padding for desired size
    # rows/height
    num_rows_to_pad = paper_arr_rows - img_rows if paper_arr_rows > img_rows else 0
    rows_before = int(np.floor(num_rows_to_pad/2))
    rows_after = int(np.ceil(num_rows_to_pad/2))
    
    # cols/width 
    num_cols_to_pad = paper_arr_cols - img_cols if paper_arr_cols > img_cols else 0
    # cols_before = int(np.floor(num_cols_to_pad/2))
    # cols_after = int(np.ceil(num_cols_to_pad/2))
    
    
    # h5 images
    h5_data[idx,...] = idx
    channel = 0
    
    # trim height to 740
    #image = image[:paper_arr_rows, :] # pad height
    
    # pad to size of img_rows, img_cols
    # only pad front 
    image = np.pad(image,  # pad width
                   ((rows_before,rows_after),(num_cols_to_pad,0)), # this takes in width(cols) and height(rows)
                   'constant', constant_values=0 
                   )

    h5_data[idx,channel,:,:] =  image  # shift vertically
    
    # h5 labels 
    dim_class = 0
    dim_weights = 1
    
    ###
    #labels = labels[:paper_arr_rows, :] # trim height
    top_layer = int(np.unique(labels)[-1]) # last layer value
    bottom_layer = int(np.unique(labels)[0]) # first layer
    labels = np.pad(labels,  # pad width
       ((rows_before,rows_after),(num_cols_to_pad,0)),
       'constant', constant_values=((0,0),(bottom_layer,top_layer)) # fill with area above retina 
       )

    weights = weights[:paper_arr_rows, :] # trim height
    weights = np.pad(weights,  # pad width
       ((rows_before,rows_after),(num_cols_to_pad,0)), #pad at front only
       'constant', constant_values= 1  # fill with bg value 
       )

    ###
    
    h5_labels[idx, dim_class, :, :] = labels # pad to size of img_rows, img_cols
    h5_labels[idx,dim_weights, :, :] = weights  # pad to size of img_rows, img_cols
    



# split dataset
percent_class_training = 0.8 # class 1 = training
class_validation = 3
h5_set[int(num_images*percent_class_training):, 0] = class_validation
    
#EXPORT DATASETS

# output_path = Path(__file__).parent.absolute() # Path("/home/skalalab/Desktop/relaynet_unmodified/relaynet_pytorch-master")


# #labels should start with 1
# for label in h5_labels:
#     print(np.unique(label[0,...]))
    
with h5py.File(f"{str(path_h5_output / 'data_w_augs.h5')}", 'w') as f:
    f.create_dataset('oct_data', data = h5_data.astype(np.float64, copy=False))
with h5py.File(f"{str(path_h5_output / 'labels_w_augs.h5')}", 'w') as f:
    f.create_dataset('oct_labels', data = h5_labels.astype(np.float64, copy=False))  
with h5py.File(f"{str(path_h5_output / 'set_w_augs.h5')}", 'w') as f:
    f.create_dataset('oct_set', data = h5_set.astype(np.float64, copy=False))  

print(f"output directory: {path_h5_output}")

####################################################################
# fix filenames in May's images 
# roi_path = Path("C:/Users/econtrerasguzman/Desktop/fix")

# roi_path = roi_path / "2019_03_06_human_amniochorion_labored_term_AROM_periplacental_0002_Mode2D" / "roi_files"
 

# list_files= list(roi_path.glob("*.zip"))
# for path_file in list_files:


#     filename = path_file.name
#     filename = filename.split(sep="_")
    
#     output_filename = "_".join(filename[:-2])
#     output_filename = "_".join([output_filename, filename[-1]])
    
#     os.rename(path_file, str(path_file.parent / output_filename))

