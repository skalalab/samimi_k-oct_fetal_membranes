# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:02:19 2021

@author: OCT
"""
from flim_tools.metrics import dice, jaccard
from flim_tools.visualization import compare_orig_mask_gt_pred

# load h5 files
# separate h5 test files
# get masks
# with model predict on image:
#     output --> compare with mask_gt per layer --> (dice, jaccard, total error)
    
import numpy as np
import matplotlib.pyplot as plt
import torch
# from torch.autograd import Variable
from pathlib import Path
#ECG: you will also need to install h5py in the conda env
import os 

import pandas as pd

import torch.nn.functional as F

import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300


# change directory to relaynet folder
os.chdir(r"C:\Users\OCT\Desktop\development\relaynet_pytorch".replace("\\", "/"))

# on docker
# os.chdir("/tmp/relaynet/relaynet_pytorch-master")

# from relaynet_pytorch.relay_net import ReLayNet
from relaynet_pytorch.data_utils import get_imdb_data
# from relaynet_pytorch.relay_net import ReLayNet
# from relaynet_pytorch.solver import Solver
# import relaynet_pytorch


    
#%% LOAD DATA

fold = "fold_0"
suffix= f"_w_augs_{fold}"
path_dataset =  Path(f"F:/Emmanuel/0-h5/{fold}")


## small dataset 
# fold = "fold_0"
# suffix= f"_w_augs_{fold}_small"
# #local path
# path_dataset = r"Z:\0-Projects and Experiments\KS - OCT membranes\relaynet_small_dataset".replace("\\","/")
#Docker path
# path_dataset = Path("/tmp/dataset")

path_dataset =  Path(path_dataset)

rows_slicing, cols_slicing = (50,-50), ("start", "end")
train_data, test_data = get_imdb_data(path_dataset, suffix, row_slice=rows_slicing, col_slice=cols_slicing) #,row_upper_limit,column_lower_limit )

### ECG objects should have format: 
# X = (176, 1, 435, 768) [num_image, channel, rows, cols]
# y (176, 435, 768) [num_image, rows, cols]
# w (176, 10, 435, 768) [num_image, num_layers, rows, cols]
print(f"train X : {train_data.X.shape}")
print(f"train y : {train_data.y.shape}")
print(f"train w : {train_data.w.shape}")
###

print("Train size: %i" % len(train_data))
print("Test size: %i" % len(test_data))

    
#%%   segment test samples 
    
      

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



#%%

from skimage.color import label2rgb
from flim_tools.image_processing import normalize
from flim_tools.metrics import dice, total_error


def compare_orig_mask_gt_pred(im, mask_gt, mask_pred, title="" ):
    alpha = 0.5
    im_overlay = label2rgb(mask_pred, normalize(im), bg_label=0, alpha=alpha, image_alpha=1, kind="overlay")
    
    fig, ax = plt.subplots(1,5, figsize=(10,7))
    
    plt.suptitle(title)
    ax[0].title.set_text(f"original")
    ax[0].set_axis_off()
    ax[0].imshow(im)
    
    # overlayed 
    dice_coeff = dice(mask_pred, mask_gt) 
    ax[1].title.set_text(f"overlayed mask_pred")
    ax[1].set_axis_off()
    ax[1].imshow(im_overlay)
    
    # mask gt 
    ax[2].title.set_text(f"mask_gt")
    ax[2].set_axis_off()
    ax[2].imshow(mask_gt)
    
    # mask pred
    ax[3].title.set_text(f"mask_pred \n dice: {dice_coeff:.4f}")
    ax[3].set_axis_off()
    ax[3].imshow(mask_pred)
    
    ## XOR
    mask_xor = np.logical_xor(mask_gt,mask_pred)
    
    error_total = total_error(mask_gt, mask_pred)
    ax[4].title.set_text(f"mask_xor\n total error: {(error_total):.3f}")
    ax[4].set_axis_off()
    ax[4].imshow(mask_xor)
    
    plt.show()
    
#%%  calculate metrics per image

# img_num = 11

dict_layers = {
    1: "decidua",
    2: "chorion",
    3: "spongy",
    4: "amnion"
    }

dict_metrics = {}

# iterate through frames
for img_num in np.arange(len(test_data)):
    pass
    print(f"processing image {img_num+1}/{len(test_data)} ")

    # add key
    dict_metrics[img_num] = {}

    with torch.no_grad(): # this frees up GPU memory in between runs!!!
        
        # print(test_data.X[img_num:img_num+1,...].shape)
    
        # path_model = Path("Z:/0-Projects and Experiments/KS - OCT membranes/trained_models/relaynet_model_fold_9.model")
        # path_model = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\relaynet_small_dataset\relaynet_model_fold_0.model".replace("\\",'/'))
        path_model = Path(r"F:\Emmanuel\0-h5\fold_0\relaynet_model_fold_0.model".replace("\\",'/'))
        
        relaynet_model =  torch.load(str(path_model))
        #out = relaynet_model(Variable(torch.Tensor(test_data.X[0:1]).cuda(),volatile=True)) # originally 
        out = relaynet_model(torch.Tensor(test_data.X[img_num:img_num+1,...]).cuda())
        out = F.softmax(out,dim=1)
        max_val, idx = torch.max(out,1)
        idx = idx.data.cpu().numpy()
        # idx = label_img_to_rgb(idx) # comment in to color image
        idx = np.squeeze(idx) #ECG added 
        # print(np.unique(idx), idx.shape)
        # plt.imshow(idx == 1) # show only one layer
        # plt.imshow(idx)
        # plt.show()
        
        # show original image
        # img_test = test_data.X[img_num:img_num+1,...] 
        # img_test = np.squeeze(img_test)
        # plt.imshow(img_test)
        # plt.show()
        
        # remove background layers at start/end
        list_layers = np.unique(idx)[1:-1]
        

        # iterate through layers and compare with gt mask
        for layer_num in list_layers:
            pass

            layer_name = dict_layers[layer_num] # get layer name
            
            # make dict to store layer metrics
            dict_metrics[img_num][layer_name] = {}
            
            # get images and crop to original height of 265
            im = test_data.X.squeeze()[img_num,...][:,-265:] # original 
            mask_gt = (test_data.y[img_num,...] == layer_num)[:,-265:] # ground truth
            mask_pred = (idx == layer_num)[:,-265:] # mask_pred 
            
            ## compute and save metrics
            dict_metrics[img_num][layer_name]["dice"] = dice(mask_gt, mask_pred)
            dict_metrics[img_num][layer_name]["jaccard"] = jaccard(mask_gt, mask_pred)
            dict_metrics[img_num][layer_name]["total_error"] = total_error(mask_gt, mask_pred)
            

            compare_orig_mask_gt_pred(im.copy(), mask_gt.copy(), mask_pred.copy(), title=f"{layer_name}, test_sample: {img_num}")
    
df = pd.DataFrame(dict_metrics)
#%% plots

# iterate through indices/layers
for ind in df.index:
    pass
    list_dice = []
    list_jaccard = []
    list_total_error = []
    
    #  iterate through all samples
    for sample_num in df.keys():
        pass

        list_dice.append(df[sample_num][ind]["dice"])
        list_jaccard.append(df[sample_num][ind]["jaccard"])
        list_total_error.append(df[sample_num][ind]["total_error"])
    
    
    fig, ax = plt.subplots(1,2, figsize=(20,6))
    fig.suptitle(f"{ind}")
    
    ax[0].set_title(f"dice \n mean: {np.mean(list_dice):.3f}  stdev: {np.std(list_dice):.3f}")
    ax[0].plot(list_dice)
    ax[0].set_xlabel("Sample index ")
    ax[0].set_ylabel("Dice Score")
    # ax[1].set_title(f"jaccard \n mean: {np.mean(list_jaccard):.3f}  stdev: {np.std(list_jaccard):.3f}")
    # ax[1].plot(list_jaccard)

    ax[1].set_title(f"total_error \n mean: {np.mean(list_total_error):.3f}  stdev: {np.std(list_total_error):.3f}")
    ax[1].plot(list_total_error)
    ax[1].set_xlabel("Sample index ")
    ax[1].set_ylabel("total error (% misclassified pixels)")
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
