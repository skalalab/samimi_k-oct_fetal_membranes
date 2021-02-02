# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:38:25 2020

@author: econtrerasguzman
"""
import h5py
from pathlib import Path
import matplotlib.pylab as plt
import numpy as np

dir_dataset = Path("Z:/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/0-other_files/h5_dataset_files")


data_h5 = dir_dataset / "Data.h5"
labels_h5 = dir_dataset / "label.h5"
set_h5 = dir_dataset / "set.h5"

#reduce size of images
hf_data = h5py.File(data_h5,"r")
hf_data = hf_data.get('oct_dataset') # (265, 3100, 1, 458)
# plt.imshow(hf_data[...,0,1])
images = hf_data[...,:50] # (265, 3100, 1, 5) # slice here! last dim is img number

# reduce size of labels
hf_labels = h5py.File(labels_h5,"r")
hf_labels = hf_labels.get('oct_labels')
hf_labels.shape # (265, 3100, 2, 458)
# plt.imshow(hf_labels[...,0,1])
labels = hf_labels[...,:50] # slice here! last dim is img number

# reduce size of set
hf_set = h5py.File(set_h5,"r")
hf_set = hf_set.get('Set')
hf_set.shape # (1, 458)
# plt.imshow(hf_d[...,0,1])
Set = hf_set[...,:50] # slice here! last dim is img number
_, num_imgs = Set.shape
start =  int(num_imgs*0.8)
Set[0,start:] = 3

# h5_set[int(num_images*percent_class_training):, 0] = class_validation


##shape for ECG relaynet
images = np.swapaxes(images,3,2)
images = np.swapaxes(images,2,1)
images = np.swapaxes(images,1,0)
images = np.swapaxes(images,3,2)
images = np.swapaxes(images,2,1)
images = np.swapaxes(images,3,2)

labels = np.swapaxes(labels,3,2)
labels = np.swapaxes(labels,2,1)
labels = np.swapaxes(labels,1,0)
labels = np.swapaxes(labels,3,2)
labels = np.swapaxes(labels,2,1)
labels = np.swapaxes(labels,3,2)

Set = np.swapaxes(Set, 1,0)


with h5py.File(str(dir_dataset / "ecg_small_dataset"/"set_ecg.h5"), 'w') as f:
    f.create_dataset('Set', data = Set) 
    
with h5py.File(str(dir_dataset / "ecg_small_dataset" / "data_ecg.h5"), 'w') as f:
    f.create_dataset('oct_dataset', data = images)  
    
with h5py.File(str(dir_dataset / "ecg_small_dataset" / "labels_ecg.h5"), 'w') as f:
    f.create_dataset('oct_labels', data = labels)  