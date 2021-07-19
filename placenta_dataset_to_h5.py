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
# import random
import h5py
import skimage.transform
# from skimage.draw import line
# from read_roi import read_roi_zip

# in spyder change figures to show higher, otherwise it shows gaps in layers
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400

####
from sklearn.model_selection import KFold

#linux 
# path_segmented_dataset = Path("/run/user/1000/gvfs/smb-share:server=skala-dv1.discovery.wisc.edu,share=ws/skala/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/0-segmentation_completed")
# Windows
# path_segmented_dataset = Path("Z:/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/0-segmentation_completed")
# Windows on desktop 
path_segmented_dataset = Path("F:/Emmanuel/0-segmentation_completed")

assert path_segmented_dataset.exists() == True, print(f"path_segmented_datset doesn't exists: {path_segmented_dataset.exists()}")

path_h5_output = path_segmented_dataset.parent / "0-h5"

# path_output = Path("Z:\0-Projects and Experiments\KS - OCT membranes\oct_dataset_3100x256\0-segmented_dataset")

list_images = []
list_labels = []
list_weights = []
not_found = 0

# store paths to images to do the cross validation
list_path_frame = []
list_path_rois = []
# path_roi = Path("/home/skalalab/Desktop/test_roi_set.zip")
# rois = read_roi_zip(path_roi) 

# iterate through each data folder
list_sample_dirs = list(path_segmented_dataset.glob("*_amniochorion_*"))
for img_folder in list_sample_dirs: # [0:1] # get first image 
# for img_folder in [list(path_segmented_dataset.glob("*_amniochorion_*"))[-1]]: # get last image
    pass
    
    print(f"***** Processing Directory: {img_folder.name}")
    path_images = img_folder / "images"
    path_rois = img_folder / "roi_files"
    
    # iterate through each roi
    for path_roi_file in list(path_rois.glob("*.zip")):
        pass
        print(f"roi found: {path_roi_file.name}")
        
        # populate list of paths to images
        list_path_rois.append(path_roi_file)
        

        # image path
        path_image = path_images / f"{path_roi_file.stem}.tiff"
        
        # validate paths
        # suspect too long image paths raise FileNotFound exception
        try:
            if path_roi_file.exists() == False :
                print("roi file not found")
                raise FileNotFoundError
            
            if path_image.exists() == False :
                print("image file not found")
                raise FileNotFoundError
        except:
            not_found += 1
            print(f"file not found: {str(path_roi_file.name)}")
            continue
        
        # add valid path to image
        list_path_frame.append(path_image)
        
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
        
        
        # check that  labels have correct number of layers
        assert len(np.unique(mask_labels)) == 6, f"Error: incorrect number of labels in mask {len(np.unique(mask_labels_encoded))} | {path_roi_file} \n likely rois in imagej are not arraged top to bottom layer"
        # if len(np.unique(mask_labels)) != 6:
        #     print(f"======> Error: incorrect number of labels in mask {len(np.unique(mask_labels_encoded))} | {path_roi_file}")

        
        # plt.imshow(mask_labels)
        # plt.show()
        
        # get weights mask
        w1 = 10
        w2 = 50
        mask_weights = ru.generate_weights_oct(mask_labels, w1, w2)
        list_weights.append(mask_weights)
        # plt.imshow(mask_weights)
        # plt.show()


# make a list of image/mask paths 
# split into training/validation

print(f"number of images: {len(list_images)}")
print(f"number of labels masks: {len(list_labels)}")
print(f"number of weight masks: {len(list_weights)}")
print("-" * 20)
print(f"number of paths to frames: {len(list_path_frame)}")
print(f"number of paths to rois: {len(list_path_rois)}")
print("-" * 20)
print(f"images not found: {not_found}")


## get list of dict
list_list_frame_set = []
for im, m_labels, m_weights, p_frame, p_rois in list(zip(list_images, list_labels,list_weights, list_path_frame, list_path_rois)):
    dict_set = {
        "image": im,
        "m_labels": m_labels,
        "m_weights": m_weights,
        "path_frame": p_frame,
        "path_rois": p_rois
        }
    list_list_frame_set.append(dict_set)
                                                     

#%% splitting into test/train here

kf = KFold(n_splits=10, shuffle=True,  random_state=33)

dict_kfold_indices = {} # dictionary of fold indices based split by cross validation

#split fetal membrane samples into test/train
for idx, (train_indices, test_indices) in enumerate(list(kf.split(list_sample_dirs))):
# for idx, (train_indices, test_indices) in enumerate(list(kf.split(list_list_frame_set))):

    print("-" * 50)
    print(f"fold: {idx}")
    print(f"train: {train_indices}")
    print(f"test: {test_indices}")
    print("-" * 50)
    dict_kfold_indices[f"fold_{idx}"] = {
        "train": train_indices,
        "test": test_indices
        }

# split samples  according to folds
dict_folds = {} # holds cross validation datasets

## iterate through 
for idx, fold in enumerate(dict_kfold_indices.keys()):
    pass

    ## PACK TRAINING
    list_train_indices =  dict_kfold_indices[fold]["train"]
    
    list_train = [] # list to store samples
    for sample_dir_idx in list_train_indices:
        pass
        sample_dir = list_sample_dirs[sample_dir_idx]
        # compare frame_folder to sample folder
        list_train += [frame_dict for frame_dict in list_list_frame_set if frame_dict["path_frame"].parent.parent.name == sample_dir.name]
    
    ## PACK TESTING
    list_test_indices = dict_kfold_indices[fold]["test"]
    list_test = [] # list to store samples
    for sample_dir_idx in list_test_indices:
        pass
        sample_dir = list_sample_dirs[sample_dir_idx]
        # copy testing only for specific samples
        list_test += [frame_dict for frame_dict in list_list_frame_set if frame_dict["path_frame"].parent.parent.name == sample_dir.name]
    
    dict_folds[f"{fold}"] = {
        "train_set": list_train,
        "test_set" : list_test
        }

#%% Apply augmentations
# store augmentations for each fold
dict_fold_augs = {}

for fold_idx, fold in enumerate(list(dict_folds.keys())[3:]): # [0] export first  model
# for fold_idx, fold in enumerate(dict_folds.keys()):
    print(f"Packaging {fold}:  {fold_idx+1}/{len(dict_folds.keys())}")
    pass
    debug = False
    # load subset of folds 
    list_images = [train_set["image"] for train_set in dict_folds[fold]["train_set"]]
    list_labels = [train_set["m_labels"] for train_set in dict_folds[fold]["train_set"]]
    list_weights = [train_set["m_weights"] for train_set in dict_folds[fold]["train_set"]]
    
    
    # list_image_name 
    list_path_frame = [train_set["path_frame"] for train_set in dict_folds[fold]["train_set"]]
    
    ## export small dataset: 
    # list_images = list_images[:10]
    # list_labels = list_labels[:10]
    # list_weights = list_weights[:10]
    
    # lists that store augmentations
    list_images_aug = []
    list_labels_aug = []
    list_weights_aug = []
    
    # do augmentations 
    for pos, (image, mask_labels, mask_weights, path_frame) in enumerate(zip(list_images, list_labels, list_weights, list_path_frame)): # select a subset
        pass
        print(f"Augmenting image: {pos+1}/{len(list_images)} | {path_frame.name}")
              
        ##### MIRROR DATA

        im_mirrored = ru.mirror_array(image, show_image=debug)
        labels_mirrored = ru.mirror_array(mask_labels, show_image=debug)
        weights_mirrored = ru.mirror_array(mask_weights, show_image=debug)
        
        if debug:
            plt.title("mirrored")
            plt.imshow(im_mirrored)
            plt.show()
            plt.title("mirrored")
            plt.imshow(labels_mirrored)
            plt.show()
            plt.title("mirrored")
            plt.imshow(weights_mirrored)
            plt.show()
        
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
            if debug:
                plt.title("translate")
                plt.imshow(t_im)
                plt.show()
                plt.title("translate")
                plt.imshow(t_labels)
                plt.show()
                plt.title("translate")
                plt.imshow(t_weights)
                plt.show()
                
       
        ##### SCALE DATA
        num_rows, num_cols = image.shape
        resize_percent = 1.1
        scaled_rows_size = int(num_rows*resize_percent)
        scaled_cols_size = int(num_cols*resize_percent)
        
        
        im_scaled = skimage.transform.resize(image,(scaled_rows_size, scaled_cols_size), preserve_range = True, anti_aliasing=False, order=0)
        labels_scaled = skimage.transform.resize(mask_labels,(scaled_rows_size, scaled_cols_size), preserve_range = True, anti_aliasing=False, order=0)
        weights_scaled = skimage.transform.resize(mask_weights,(scaled_rows_size, scaled_cols_size), preserve_range = True, anti_aliasing=False, order=0)
        
        if debug:
            plt.title("scaled")
            plt.imshow(im_scaled)
            plt.show()
            plt.title("scaled")
            plt.imshow(labels_scaled)
            plt.show()
            plt.title("scaled")
            plt.imshow(weights_scaled)
            plt.show()
    
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
            
                   
            if debug:
                plt.title("refocusing augmentation")
                plt.imshow(rolled_image)
                plt.show()
                plt.title("refocusing augmentation")
                plt.imshow(rolled_labels)
                plt.show()
                plt.title("refocusing augmentation")
                plt.imshow(rolled_weights)
                plt.show()
        
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

    
    ##%% combine training and testing into the same array for packaging 
    
    # load ***TEST*** subset of images
    list_images_test = [test_set["image"] for test_set in dict_folds[fold]["test_set"]]
    list_labels_test = [test_set["m_labels"] for test_set in dict_folds[fold]["test_set"]]
    list_weights_test = [test_set["m_weights"] for test_set in dict_folds[fold]["test_set"]]
    
    

    # add augmetnations to training set
    list_images += list_images_aug 
    list_labels +=  list_labels_aug
    list_weights +=  list_weights_aug
    
    # save these for H5_set creation split
    n_training  = len(list_images)  # first n images are for training
    
    # append testing images 
    list_images += list_images_test
    list_labels += list_labels_test 
    list_weights += list_weights_test

    
    ## FROM RELAYNET SEGMENTATION
    #  entries 1 or 3 indicating which data is for training and validation respectively.
    dim_train_test_id = 1
    h5_set = np.zeros((len(list_images), dim_train_test_id)) # (image_number, train_test_id)
    
    class_validation = 3
    class_training = 1
    h5_set[:n_training, 0] = class_training
    h5_set[n_training:, 0] = class_validation
    
    
    ##%% instantiate data arrays
    num_images = len(list_images)
                 
    # paper stated these dimensions, pad to this below
    paper_arr_rows, paper_arr_cols = (3100,512)
                
    # imdb.images.data is a 4D matrix of size: [height, width, color channel (1ch), NumberOfData]
    image_color_channels = 1 #1 channel so grayscale
    h5_data = np.zeros((num_images, image_color_channels, paper_arr_rows, paper_arr_cols))
    
    # imdb.images.labels is a 4D matrix of size: [height, width, 2, NumberOfData] 
    dim_class_and_weights = 2 # ---> 1st Channel is class (1,2,... etc), 2nd channel is Instance Weights (All voxels with a class label is assigned a weight, details in paper)
    h5_labels = np.zeros((num_images, dim_class_and_weights, paper_arr_rows, paper_arr_cols))
    

    
    ##%%

    list_all = list(zip(list_images, list_labels, list_weights))
    
    # iterate through the number of images and make h5 stacks
    for idx, (image, labels, weights) in enumerate(list_all):
        pass
        print(f"Adding image to h5 stack {idx+1}/{len(list_all)}")
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
        # trim height to 740
        #image = image[:paper_arr_rows, :] # pad height
        
        num_cols_to_pad = paper_arr_cols - img_cols if paper_arr_cols > img_cols else 0
        # cols_before = int(np.floor(num_cols_to_pad/2))
        # cols_after = int(np.ceil(num_cols_to_pad/2))
        
        
        # h5 images
        h5_data[idx,...] = idx
        channel = 0
        
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
        
##%%
    #EXPORT DATASETS
    print("saving datasets to h5")
    # make folder to store fold
    path_h5_output_subfolder = path_h5_output / f"{fold}"
    path_h5_output_subfolder.mkdir(exist_ok=True)
    
    filename = path_h5_output_subfolder / f"data_w_augs_{fold}.h5"
    
    with h5py.File(str(filename), 'w') as f:
        f.create_dataset('oct_data', data = h5_data.astype(np.float64, copy=False))
    
    filename = path_h5_output_subfolder / f"labels_w_augs_{fold}.h5"
    with h5py.File(str(filename), 'w') as f:
        f.create_dataset('oct_labels', data = h5_labels.astype(np.float64, copy=False))  
        
    filename = path_h5_output_subfolder / f"set_w_augs_{fold}.h5"  
    with h5py.File(str(filename), 'w') as f:
        f.create_dataset('oct_set', data = h5_set.astype(np.float64, copy=False))  
    
    print(f"output directory: {path_h5_output}")


