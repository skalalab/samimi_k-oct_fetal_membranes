import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from pathlib import Path, PurePosixPath, PureWindowsPath
import numpy as np
import tifffile
import matplotlib
# matplotlib.use("Agg")
import matplotlib.animation as animation
from skimage.morphology import skeletonize, dilation, remove_small_objects
# import relaynet_utils as ru
import cv2
from skimage.draw import line
from skimage import morphology

from skimage.measure import label, regionprops
import os
import pandas as pd
import csv

# set cwd to relaynet directory
os.chdir("C:/Users/OCT/Desktop/development/fetal_membrane_kayvan")
from layer_edge_fitting_code import compute_layer_thickness
from time import time



mpl.rcParams['figure.dpi'] = 300
#%%

# # images in completed
# base_path = Path("Z:/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/0-segmentation_completed")
base_path = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\oct_dataset_3100x256\0-segmentation_completed".replace("\\",'/'))
# path_image = base_path / "2018_11_06_human_amniochorion_labored_term_SROM_periplacental_0002_Mode2D/2018_11_06_human_amniochorion_labored_term_SROM_periplacental_0002_Mode2D.tiff"


# base_path = Path("F:/Emmanuel/0-segmentation_completed")
# list_im_paths = list(base_path.glob("*"))
# list_im_paths = [p for p in list_im_paths if p.is_dir()]

# # load list of processed samples
# path_processed_file = base_path / "processed.csv"
# list_processed_samples = csv.load(path_processed_file)

# path_image_dir = list_im_paths[1] # image number completed: [0]

# C section samples starting 
# 2020_07_16_C_section_38w6d

sample_name = "2018_10_17_human_amniochorion_labored_postterm_SROM_periplacental_0002_Mode2D"
path_sample_dir = base_path / sample_name
path_image = path_sample_dir / f"{sample_name}.tiff"
# images processed = [0]


list_images = []
list_inferences = []
list_inferences_colored = []

# LOAD IMAGE AND EXTRACT CHANNEL IF NEEDED
print("reading tiff")
path_im = str(path_image)
im_set = tifffile.imread(path_im)
print("finished reading tiff")

# account for 3 channel images, take first
if len(im_set.shape) == 4:
    im_set = im_set[..., 0]  # keep only 1 channel

# %% inferences

os.chdir("C:/Users/OCT/Desktop/development/relaynet_pytorch")

# Taken from relaynet original repo to colro images
SEG_LABELS_LIST = [
    {"id": -1, "name": "void", "rgb_values": [0, 0, 0]},
    {"id": 0, "name": "Region above the retina (RaR)", "rgb_values": [
        128, 0, 0]},
    {"id": 1, "name": "ILM: Inner limiting membrane",
        "rgb_values": [0, 128, 0]},
    {"id": 2, "name": "NFL-IPL: Nerve fiber ending to Inner plexiform layer",
        "rgb_values": [128, 128, 0]},
    {"id": 3, "name": "INL: Inner Nuclear layer", "rgb_values": [0, 0, 128]},
    {"id": 4, "name": "OPL: Outer plexiform layer",
        "rgb_values": [128, 0, 128]},
    {"id": 5, "name": "ONL-ISM: Outer Nuclear layer to Inner segment myeloid",
        "rgb_values": [0, 128, 128]},
    {"id": 6, "name": "ISE: Inner segment ellipsoid",
        "rgb_values": [128, 128, 128]},
    {"id": 7, "name": "OS-RPE: Outer segment to Retinal pigment epithelium",
        "rgb_values": [64, 0, 0]},
    {"id": 8, "name": "Region below RPE (RbR)", "rgb_values": [192, 0, 0]}]
# {"id": 9, "name": "Fluid region", "rgb_values": [64, 128, 0]}];


def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1, 2, 0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)


with torch.no_grad():  # this frees up memory in between runs!!!!
    # for im_path in path_images.glob("*.tiff"):
    pass

    # PAD IMAGE TO TRAINED MODEL DIMENSIONS
    for pos, im in enumerate(im_set):
        print(f"getting inference for image: {pos+1}/{im_set.shape[0]}")
        list_images.append(im)
        im = im.transpose()  # shift vertically
        im = im[50:-50, :]

        kernel_width = 512
        num_cols_to_pad = kernel_width - im.shape[1]  # calculate cols to pad
        im = np.pad(im,  # pad width
                    # this takes in width(cols) and height(rows)
                    ((0, 0), (num_cols_to_pad, 0)),
                    'constant', constant_values=0
                    )
        # plt.imshow(im)
        # plt.show()

        # format input
        im_rows, im_cols = im.shape
        # 1,1 are num_images, channel
        image = np.ones((1, 1, im_rows, im_cols))

        # shape input matrix
        image[0, 0, ...] = im

        
        path_model = Path(r"F:\Emmanuel\0-h5\fold_0\relaynet_model_fold_0.model".replace("\\",'/'))


        relaynet_model = torch.load(str(path_model))
        # out = relaynet_model(Variable(torch.Tensor(test_data.X[0:1]).cuda(),volatile=True)) # originally
        # out = relaynet_model(torch.Tensor(image).cuda())
        out = relaynet_model(torch.Tensor(image).cuda())
        out = F.softmax(out, dim=1)
        max_val, idx = torch.max(out, 1)
        idx = idx.data.cpu().numpy()
        # idx = label_img_to_rgb(idx) # originaly commented in
        idx = np.squeeze(idx)  # ECG added
        #print(np.unique(idx), idx.shape)
        # plt.imshow(idx == 1) # show only one layer
        # plt.imshow(idx)
        # plt.show()

        # append inference
        list_inferences.append(idx)

        list_inferences_colored.append(label_img_to_rgb(idx))
print("finished inference code")
# %%

# for im in list_inferences:
for im in list_inferences_colored[:10]:
    
    plt.imshow(im)
    plt.show()
#%%
# save labeled mask
# make tiffs
# tiff_stack = np.empty((len(list_inferences), *idx.transpose().shape))
# print("saving tiff combined output")
# out_path = str(path_image.parent / f"{path_image.stem}_combined_edges.tiff")
# with tifffile.TiffWriter(out_path, bigtiff=True) as tif: # imagej=True
#     for pos, (image, labels, edges) in enumerate(zip(list_images, list_inferences_colored)): #, edge_masks)):
#         # tiff_stack[pos,...] = labels.transpose().astype(int)
#         print(f"saving image {pos+1}/{len(list_images)}")

#         #norm_image = (image/np.max(image))[:,50:-50] # match dimensions of labels mask
#         #norm_labels = (labels.transpose()/np.max(labels)).astype(np.float32)

#         #### shape dimensions to match
#         #original image cut sides by 50
#         image = image[:,50:-50]
#         image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

#         #shape labels
#         labels = np.swapaxes(labels,0,1)
#         labels = labels[-265:,...] # take last 265 pixels

#         overlayed_im_mask = cv2.addWeighted(image,1,labels,0.5,0)
#         # plt.imshow(overlayed_im_mask)

#         #### add edges
#         # edges = edges.astype(np.float32)
#         # edges[edges > 0] = 255
#         # edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)

#         # stack images
#         # stack = np.vstack((image, overlayed_im_mask, edges[247:,:])) # ctop edges to original size (265,3000)
#         stack = np.vstack((image, overlayed_im_mask))

#         # add frame number to output image
#         stack = cv2.putText(np.array(stack), f"{pos}",(2700,100), cv2.FONT_HERSHEY_PLAIN ,5,(255,255,255), 3)
#         # plt.imshow(stack)

#         tif.save(stack.astype(np.float32))
# print(f"finished saving combined image: {out_path}")

#%% # calculate layer thickness from labels mask

import warnings
warnings.simplefilter('ignore', np.RankWarning)

#store thicknesses per frame for each layer in this dictionary
dict_layer_props = {
    "decidua": {# layer 1
        "thickness": [],
        "area": [],
        "length":[]
        },
    "chorion": {# layer 2
        "thickness": [],
        "area": [],
        "length":[]
        },
    "spongy": { # layer 3
        "thickness": [],
        "area": [],
        "length":[]
        },
    "amnion": { # layer 4
        "thickness": [],
        "area": [],
        "length":[]
        }
}


edges_decidua = []
edges_chorion = []
edges_spongy = []
edges_amnion = []
list_layer_edges = [edges_decidua, edges_chorion, edges_spongy, edges_amnion]

#TODO 
start = time() 
for frame_num, (image, labels) in enumerate(zip(list_images[:455], list_inferences), start=1):
    print(
        f"calculating layer thickness for frame: {frame_num}/{len(list_images)}")
    pass

    # iterate through each layer
    # calculate layer thickness
    # exclude top and bottom layers of placenta
    for layer_num, dict_layer_key, list_data_edges in zip((np.unique(labels)[1:-1]), dict_layer_props, list_layer_edges):
        pass
        # convert layer to ints
        layer_mask = (labels == layer_num).astype(int)

        debug = False
        # calculate layer length

        # CLEAN UP MASKS
        # layer_mask = list_inferences[12] == 1
        if debug:
            plt.title(
                f"original image: {dict_layer_key} frame {frame_num}")
            plt.imshow(image)
            plt.show()
            plt.title(
                f"original mask: {dict_layer_key} frame {frame_num}")
            plt.imshow(layer_mask.transpose())
            plt.show()

        # binary close (dialte and erode) to fill holes
        # layers are vertcal so make a vertical rectangular element
        selem_rect = morphology.rectangle(15, 5) # size to merge layer pieces if dissconected 
        layer_mask = morphology.binary_closing(
            layer_mask, selem_rect).astype(int)
        if debug:
            plt.title(
                f"after binary closing: {dict_layer_key} frame {frame_num}")
            plt.imshow(layer_mask.transpose())
            plt.show()

        ### start function for thickness calculation
        layer_thickness, mean_layer_length, layer_area, list_poly_coeffs, list_mask_pixels = compute_layer_thickness(layer_mask, method=2, debug=debug)

        # save props for each layer
        dict_layer_props[dict_layer_key]["length"].append(mean_layer_length)
        dict_layer_props[dict_layer_key]["area"].append(layer_area)
        dict_layer_props[dict_layer_key]["thickness"].append(layer_thickness)

        #save binary mask of edges to displaylater
        
        m_rows, m_cols = layer_mask.shape
        mask_edges = np.zeros((m_rows,m_cols))
        for edge in list_mask_pixels:
            pass
            x_values, y_values = edge
            for x_pixel, y_pixel in zip(x_values, y_values):
                if x_pixel < m_rows and  y_pixel < m_cols:
                    mask_edges[int(x_pixel), int(y_pixel)] = 1
        
        # plt.imshow(mask_edges, vmin=0, vmax=1)
        # plt.show()
        list_data_edges.append(mask_edges)
        
end = time()
difference = (end-start)/(60*60)
print(f"finished calculating layer thickness - elapsed time: {difference:.2f} hours")
#%% EXPORT TIFF: SAVE - RUN THIS AFTER calculating layer thickness

# combine all edges into a single mask 
edge_masks = []
for pos, (e_decidua, e_chorion, e_spongy, e_amnion) in enumerate(zip(*list_layer_edges)):
    combined_mask = e_decidua + e_chorion + e_spongy + e_amnion # combine layers into ne mask
    edge_masks.append(combined_mask)

# save labeled mask
# make tiffs of original, colored overlayed predictions and edges
# tiff_stack = np.empty((len(list_inferences), *idx.transpose().shape))
print("saving tiff overlayed output")

output_dir_path = path_image.parent / "thickness"
output_dir_path.mkdir(exist_ok=True)
out_path = str(output_dir_path /  f"{path_image.stem}_all.tiff")

# out_path = str( f"/home/skalalab/Desktop/{path_image.stem}_combined_edges.tiff")
with tifffile.TiffWriter(out_path, bigtiff=True) as tif:  # imagej=True
    for pos, (image, labels, edges) in enumerate(zip(list_images, list_inferences_colored, edge_masks), start=1):
        pass
        # tiff_stack[pos,...] = labels.transpose().astype(int)
        print(f"saving image {pos}/{len(list_images)}")

        # norm_image = (image/np.max(image))[:,50:-50] # match dimensions of labels mask
        #norm_labels = (labels.transpose()/np.max(labels)).astype(np.float32)

        # shape dimensions to match
        # original image cut sides by 50
        image = image[:, 50:-50]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


        # shape labels
        labels = np.swapaxes(labels, 0, 1)
        labels = labels[-265:, ...]  # take last 265 pixels which are the image
 
        
        overlayed_im_mask = cv2.addWeighted(image, 1, labels, 0.5, 0)
        # plt.imshow(overlayed_im_mask)

        # OVERLAY EDGES
        edges = edges.astype(np.uint8).transpose() # transpose to make horizontal #astype(np.float32) 
        edges[edges > 0] = 255
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges = edges[-265:,...] # take last 265 pixels which are the same size as original image
        edges[...,0] = 0 # blank out red channel
        edges[...,2] = 0 # blank out blue channel 
        
        overlayed_edges = cv2.addWeighted(image, 1, edges,1, 0)

        # stack images
        # crop edges to original image size (265,3000)
        stack = np.vstack((image, overlayed_im_mask, overlayed_edges))
        # stack = np.vstack((image, overlayed_im_mask))

        # add frame number to output image
        stack = cv2.putText(np.array(
            stack), f"{pos}", (2700, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)
        # plt.imshow(stack)

        tif.save(stack.astype(np.float32))
print(f"finished saving combined image: {out_path}\n")
print(f"NOTE: Remember to load the images in imagej as HYPERSTACKS")

#%% # plot layer thicknesses
for layer in dict_layer_props:
    pass
    # x_axis = np.arange(len(dict_layer_props[layer]["thickness"]),) + 1
    plt.plot(dict_layer_props[layer]["thickness"])
    # plots.append(plt.plot(thickness_data))

plt.xlabel("frame number")
plt.ylabel("thickness (pixels)")
plt.title(f"{path_image.stem}")
plt.legend(list(dict_layer_props.keys()), loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
path_fig_output = output_dir_path / \
    f"{path_image.stem}_thickness.jpeg"
print(f"figure saved in: {str(path_fig_output)}")
plt.savefig(str(path_fig_output), bbox_inches="tight")
plt.show()

#%% save thickness as csv

# 2d Dict to DF
df_props = pd.DataFrame()
for layer_name in dict_layer_props:
    for prop in dict_layer_props[layer_name]:
        df_props[f"{layer_name}-{prop}"] = dict_layer_props[layer_name][prop]
        
# df_layer_thickness = pd.DataFrame(dict_layer_props["amnion"]) # columns=["decidua","chorion", "spongy", "amnion"]


## add totals
df_props["total_thickness"] = df_props["decidua-thickness"] + \
                            df_props["chorion-thickness"] + \
                            df_props["spongy-thickness"] + \
                            df_props["amnion-thickness"]

path_csv_output = output_dir_path / \
    f"{path_image.stem}_thickness.csv"

print(f"layer thickness output path: {path_csv_output}")
df_props.to_csv(path_csv_output, index=False) # don't save row index
#%% merge matlab outputs and python into one csv file

from pathlib import Path
import pandas as pd


path_sample = path_image.parent

# path_sample = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\oct_dataset_3100x256\0-segmentation_completed\2018_10_09_human_amniochorion_labored_term_AROM_pericervical_0002_Mode2D".replace("\\",'/'))
# load matlab export csv
path_matlab_csv = list(path_sample.glob("*_Apex.csv"))[0]
df_frame_apex_rise_pressure = pd.read_csv(path_matlab_csv, names=["Apex Rise", "Pressure"])

# load python thickness
path_frame_vs_thickness = path_sample / "thickness"
path_frame_vs_thickness_csv = list(path_frame_vs_thickness.glob("*.csv"))[0]
df_frame_thickness = pd.read_csv(path_frame_vs_thickness_csv)


### check that they are the same length, extend if necessary
if len(df_frame_apex_rise_pressure) != len(df_frame_thickness):
    # this is used in the event that not all frames are segmented by relaynet
    # if for some reason the membrane goes out of screen and segmentation fails
    print("Error: matlab and python export don't match in number of frames")
    print(f"apex df: {len(df_frame_apex_rise_pressure)} | thickness df: {len(df_frame_thickness)}")
    answer = input("Extend thickness df to match apex rise df? (y/n)")
    if answer == "y":
        ## fill thickness df to match apex rise
        num_missing_rows= len(df_frame_apex_rise_pressure) - len(df_frame_thickness)
        df_copy = df_frame_thickness.copy()
        # create list of indices to match apex rise df
        list_new_indices = np.arange(num_missing_rows) + len(df_frame_thickness)
        # print(df_copy.loc[len(df_copy.index)-1])
        filler_row = ["NA"] * len(df_frame_thickness.columns) # create filler row matching number of cols 
        for idx in list_new_indices: #fill indicex with filler row
            df_copy.loc[idx] = filler_row
        df_frame_thickness = df_copy # replace df
    else:
        assert len(df_frame_apex_rise_pressure) == len(df_frame_thickness), "df_apex is not the same length as df_thickness"
# merge dataframes
df_merged = pd.concat([df_frame_apex_rise_pressure, df_frame_thickness], axis=1)

# reindex rowsto star at 1
df_merged.index = np.arange(1, len(df_frame_thickness)+1)
df_merged.index.names = ["frame_number"] 

path_features_csv = path_sample / f"{path_sample.name}_features.csv"
df_merged.to_csv(path_features_csv, na_rep="NA")
print(path_features_csv)
#%% Copy over pressure files to hand segmented samples

# import re
# import shutil

#get list of all samples
# path_segmented = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\oct_dataset_3100x256\0-segmentation_completed".replace("\\",'/'))
# list_path_dir_segmented_samples = list(path_segmented.glob("*"))


# #get list of pressure files - do this once here
# path_pressure = Path("Z:/Kayvan/Human Data")
# list_pressure_files = list(path_pressure.rglob("*_Pressure*.txt"))
# list_pressure_files_str = [str(p) for p in list_pressure_files]


# for pos, path_dir_sample in enumerate(list_path_dir_segmented_samples):
#     pass
#     sample_name = path_dir_sample.name

#     # find corresponding pressure file that matches sample
#     path_pressure_file = list(filter(re.compile(f"{sample_name}").search, list_pressure_files_str))[0]
#     path_pressure_file = Path(path_pressure_file)
#     print(f"{pos}: {path_pressure_file.name}")
    
    # commment in to copy files
    #shutil.copy2(path_pressure_file, path_dir_sample / path_pressure_file.name)
    
    
