#!/usr/bin/env python3

from scipy.io import loadmat
import matplotlib.pylab as plt
import numpy as np

import h5py
from pathlib import Path
import random
import sys
import tempfile
import shutil
import relaynet_utils as ru

#%%
if __name__ == '__main__': 
    
     
    #check for file input
    if len(sys.argv[1:]) == 0:
        print("please enter the path to the duke dataset as first argument")
        sys.exit(1)
    else:
        path_directory = Path(sys.argv[1])
        if not path_directory.exists():
            print("Invalid directory: {str(path_directory)}")
            sys.exit(1)
            
    
    path_directory = Path("C:/Users/econtrerasguzman/Desktop/data/relaynet/2015_BOE_Chiu.zip")
    
    # do everything in temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        print(temp_dir.exists())
        
        shutil.unpack_archive(path_directory, extract_dir=temp_dir)
        
        temp_dir = temp_dir / "2015_BOE_Chiu"
        
        print("files found:")
        for file in temp_dir.iterdir():
            print(file)
        
        # dataset_dir = Path("C:/Users/econtrerasguzman/Desktop/data/relaynet/2015_BOE_Chiu.zip")
        # file_dir = '/home/skalalab/Desktop/relaynet_pytorch-master/datasets'
        # file_dir = '/Users/maywu/Desktop/Skala_lab/placenta_segmentation/datasets'
        dataset_dir = temp_dir
        
        # matlab struct headers
        
        # DME is diabetic macular edema
        # struct_headers = [    
        #     "automaticFluidDME", # (496,768,61)
        #     "automaticLayersDME", # (rows, cols, num_img )
        #     "automaticLayersNormal", # (8,768,61)
        #     "images", # (496,768,61)
        #     "manualFluid1", # (496,768,61) # nan's, and fluid 
        #     "manualFluid2", # (496,768,61)
        #     "manualLayers1", # (8,768,61)
        #     "manualLayers2" # (8,768,61)
        # ]
        
        #export PATH="/usr/local/bin/mc:$PATH"
        # variables setup
        
        list_mat_paths = list(dataset_dir.glob("*.mat"))
        print(len(list_mat_paths))
        
        # load single mat file to get image shape
        data_struct = loadmat(list_mat_paths[0])
        img_rows, img_cols, _ = data_struct["images"].shape
        
        # OUTPUT ARRAYS
        list_images = []
        list_manual_layer_labels = []
        list_weights = []
            
        num_images = 0 # keep track of number of images in dataset
    
        for mat_file_path in list_mat_paths:
            
            #mat_file_path = list_mat_paths[0]# load 1st image for testing 
            
            # LOAD MAT STRUCT 
            mat_struct = loadmat(mat_file_path) # previously dictionary
            
            # get index of segmented images
            valid_frames, valid_range = ru.get_annotated_img_index(mat_struct['manualLayers1'])
            
    
            # iterate through this struct and grab masks from each
            # segmenter
            for segmenter in [1,2]: # iterate through both drs
                for valid_frame in valid_frames:
                    num_images += 1 # keep track of number of images
                    list_images.append(mat_struct["images"][...,valid_frame])
                    
                    preview_images = False
                    
                    if preview_images: # make true to show image
                        plt.imshow(mat_struct["images"][...,valid_frame])
                        plt.show()
                    
                    # generate labeled mask
                    mask_layers = ru.generate_labels_mask(mat_struct[f"manualLayers{segmenter}"][..., valid_frame], img_rows, img_cols, show_mask=preview_images)
                    
                    # make nans into - (void layer)
                    mask_layers[np.isnan(mask_layers)] = -1
                    
                    # Add fluid layers
                    mask_fluid = mat_struct[f"manualFluid{segmenter}"][...,valid_frame]
                    mask_layers[mask_fluid > 0] = -1
                    
                    ### offset layers 
                    ### relaynet expects layers to be 1-num layers because they subtract 1 to have layers be 0-(num_layers-1) 
                    # my layers are -1 for void then 0 to (num_layers), offset everything by 2 so labels start at 1
                    mask_layers += 2
                    
                    #print(np.unique(mask_layers))
                    
                
                    if preview_images: # make true to show image
                        plt.imshow(mask_layers)
                        plt.show()
                        plt.imshow(mat_struct[f"manualFluid{segmenter}"][...,valid_frame])
                        plt.show()
                    
                    list_manual_layer_labels.append(mask_layers) 
                    
                    # TODO 
                    # generate the weights for the labels
                    # see paper for weight generation
                    weight_1 = 10 # for layers 
                    weight_2 = 50 # for transition pixels
                    weights = ru.generate_weights(mask_layers,weight_1 ,weight_2, show_weights_mask = preview_images)
                    list_weights.append(weights)
                    
        # end of list generation 
                
        # CREATE 4D LABELS
        # Note: github repo description is wrong for arrangement of 4D arrays
        # for relaynet setup, see below for correct arragement
        # data = [img_number, image_color_channel, rows, cols]
        # labels = [img_number, class_or_weight, rows, cols]
        # set = [num images, training_or_test_id]
                    
        # paper stated these dimensions, pad to this below
        paper_arr_rows, paper_arr_cols = (740,512)
                    
        # imdb.images.data is a 4D matrix of size: [height, width, color channel (1ch), NumberOfData]
        image_color_channels = 1
        # h5_data = np.zeros((img_rows, img_cols, image_color_channels, num_images))
        h5_data = np.zeros((num_images, image_color_channels, paper_arr_rows, paper_arr_cols))
        
        # imdb.images.labels is a 4D matrix of size: [height, width, 2, NumberOfData] 
        dim_class_and_weights = 2 # ---> 1st Channel is class (1,2,... etc), 2nd channel is Instance Weights (All voxels with a class label is assigned a weight, details in paper)
        # h5_labels = np.zeros((img_rows, img_cols, dim_class_and_weights, num_images))
        h5_labels = np.zeros((num_images, dim_class_and_weights, paper_arr_rows, paper_arr_cols))
        
        # imdb.images.set is [1,NumberOfData] vector with entries 1 or 3 indicating which data is for training and validation respectively.
        # h5_set = np.zeros((1, num_images))
        # ID for training or testing
        dim_train_test_id = 1
        h5_set = np.ones((num_images, dim_train_test_id))
    
            
        # shuffle lists
        # RANDOMIZE!
        list_all = list(zip(list_images, list_manual_layer_labels, list_weights))
        random.shuffle(list_all)    
        
        # iterate through the number of images and make h5 stacks
        for idx, (image, labels, weights) in enumerate(list_all):
            pass
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
            cols_before = int(np.floor(num_cols_to_pad/2))
            cols_after = int(np.ceil(num_cols_to_pad/2))
            
            
            # h5 images
            h5_data[idx,...] = idx
            channel = 0
            
            # trim height to 740
            image = image[:paper_arr_rows, :] # pad height
            
            # pad to size of img_rows, img_cols
            image = np.pad(image,  # pad width
                           ((rows_before,rows_after),(cols_before,cols_after)), # this takes in width(cols) and height(rows)
                           'constant', constant_values=255 
                           )
    
            h5_data[idx,channel,:,:] =  image  # shift vertically
            
            # h5 labels 
            dim_class = 0
            dim_weights = 1
            
            ###
            labels = labels[:paper_arr_rows, :] # trim height
            top_layer = int(np.unique(labels)[-1]) # last layer value
            bottom_layer = int(np.unique(labels)[0]) # first layer
            labels = np.pad(labels,  # pad width
               ((rows_before,rows_after),(cols_before,cols_after)),
               'constant', constant_values=((0,0),(bottom_layer,top_layer)) # fill with area above retina 
               )
    
            weights = weights[:paper_arr_rows, :] # trim height
            weights = np.pad(weights,  # pad width
               ((rows_before,rows_after),(cols_before,cols_after)),
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
        
        
        #labels should start with 1
        for label in h5_labels:
            print(np.unique(label[0,...]))
            
        with h5py.File(f"{str(path_directory.parent / 'data.h5')}", 'w') as f:
            f.create_dataset('oct_data', data = h5_data.astype(np.float64, copy=False))
        with h5py.File(f"{str(path_directory.parent / 'labels.h5')}", 'w') as f:
            f.create_dataset('oct_labels', data = h5_labels.astype(np.float64, copy=False))  
        with h5py.File(f"{str(path_directory.parent / 'set.h5')}", 'w') as f:
            f.create_dataset('oct_set', data = h5_set.astype(np.float64, copy=False))  
        
    
        #### export small dataset
                # reduce dataset
        h5_data_small = h5_data[:110,...]
        h5_labels_small = h5_labels[:110,...]
        h5_set_small = h5_set[:110,...]
        h5_set_small[88:,...] = 3
        
        with h5py.File(f"{str(path_directory.parent / 'data_small.h5')}", 'w') as f:
            f.create_dataset('oct_data', data = h5_data_small.astype(np.float64, copy=False))
        with h5py.File(f"{str(path_directory.parent / 'labels_small.h5')}", 'w') as f:
            f.create_dataset('oct_labels', data = h5_labels_small.astype(np.float64, copy=False))  
        with h5py.File(f"{str(path_directory.parent / 'set_small.h5')}", 'w') as f:
            f.create_dataset('oct_set', data = h5_set_small.astype(np.float64, copy=False))
            
    
            
            