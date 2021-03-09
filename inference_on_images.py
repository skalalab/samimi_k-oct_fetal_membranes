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
from layer_edge_fitting_code import compute_layer_thickness

# set cwd to relaynet directory
os.chdir("/home/skalalab/Desktop/development/relaynet_pytorch")

# in spyder change figures to show higher, otherwise it shows gaps in layers
mpl.rcParams['figure.dpi'] = 300

# path_images = Path("Z:/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256")
# # path_images = path_images / "2018_10_09_human_amniochorion_labored_term_AROM_pericervical_0004_Mode2D"
# # path_images = path_images / "2018_10_11_human_amniochorion_labored_term_SROM_pericervical_0002_Mode2D"
# path_images = path_images / "2018_12_06_human_amniochorion_labored_term_AROM_pericervical_0003_Mode2D"
# path_images = path_images / "images"

# images in completed
# base_path = Path("Z:/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/0-segmentation_completed")
base_path = Path("/run/user/1000/gvfs/smb-share:server=skala-dv1.discovery.wisc.edu,share=ws/skala/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/0-segmentation_completed")
# path_image = base_path / ""

# path_image = base_path / "2019_03_06_human_amniochorion_labored_term_AROM_periplacental_0002_Mode2D/2019_03_06_human_amniochorion_labored_term_AROM_periplacental_0002_Mode2D.tiff"
# path_image = base_path / "2018_10_09_human_amniochorion_labored_term_AROM_pericervical_0002_Mode2D/2018_10_09_human_amniochorion_labored_term_AROM_pericervical_0002_Mode2D.tiff"
# path_image = base_path / "2018_10_09_human_amniochorion_labored_term_AROM_pericervical_0004_Mode2D/2018_10_09_human_amniochorion_labored_term_AROM_pericervical_0004_Mode2D.tiff"
# path_image = base_path / "2018_10_09_human_amniochorion_labored_term_AROM_periplacental_0002_Mode2D/2018_10_09_human_amniochorion_labored_term_AROM_periplacental_0002_Mode2D.tiff"
# path_image = base_path / "2018_11_07_human_amniochorion_labored_postterm_SROM_pericervical_0002_Mode2D/2018_11_07_human_amniochorion_labored_postterm_SROM_pericervical_0002_Mode2D.tiff"
# path_image = base_path / "2018_10_25_human_amniochorion_labored_term_AROM_pericervical_0002_Mode2D/2018_10_25_human_amniochorion_labored_term_AROM_pericervical_0002_Mode2D.tiff"
path_image = base_path / "2018_10_10_human_amniochorion_labored_postterm_AROM_pericervical_0002_Mode2D/2018_10_10_human_amniochorion_labored_postterm_AROM_pericervical_0002_Mode2D.tiff"


# path_image = base_path / "2018_10_10_human_amniochorion_labored_postterm_AROM_pericervical_0004_Mode2D/2018_10_10_human_amniochorion_labored_postterm_AROM_pericervical_0004_Mode2D.tiff"


# images not yet segmented
# path_image = Path("Z:/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/2018_10_09_human_amniochorion_labored_term_AROM_pericervical_0004_Mode2D/2018_10_09_human_amniochorion_labored_term_AROM_pericervical_0004_Mode2D.tiff")
# path_image = Path("Z:/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/2018_10_09_human_amniochorion_labored_term_AROM_periplacental_0004_Mode2D/2018_10_09_human_amniochorion_labored_term_AROM_periplacental_0004_Mode2D.tiff")
#path_image = Path("Z:/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/2018_10_11_human_amniochorion_labored_term_SROM_pericervical_0002_Mode2D/2018_10_11_human_amniochorion_labored_term_SROM_pericervical_0002_Mode2D.tiff")
# path_image = Path("Z:/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/2018_11_06_human_amniochorion_labored_term_SROM_periplacental_0003_Mode2D/2018_11_06_human_amniochorion_labored_term_SROM_periplacental_0003_Mode2D.tiff")
# path_image = PureWindowsPath("Z:/0-Projects and Experiments/KS - OCT membranes/oct_dataset_3100x256/0-segmentation_completed/2018_10_09_human_amniochorion_labored_term_AROM_pericervical_0002_Mode2D/2018_10_09_human_amniochorion_labored_term_AROM_pericervical_0002_Mode2D.tiff")

list_images = []
list_inferences = []
list_inferences_colored = []

# LOAD IMAGE AND EXTRACT CHANNEL IF NEEDED
print("reading tiff")
im_set = tifffile.imread(str(path_image))
print("finished reading tiff")

# account for 3 channel images, take first
if len(im_set.shape) == 4:
    im_set = im_set[..., 0]  # keep only 1 channel

# %% inferences

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

        # path_model = Path("Z:/0-Projects and Experiments/KS - OCT membranes/trained_models/relaynet_model.model")
        path_model = Path(
            "Z:/0-Projects and Experiments/KS - OCT membranes/trained_models/relaynet_model_w_augs.model")
        # path_model = Path("/home/skalalab/Desktop/relaynet_model_w_augs.model")
        path_model = Path(
            "/run/user/1000/gvfs/smb-share:server=skala-dv1.discovery.wisc.edu,share=ws/skala/0-Projects and Experiments/KS - OCT membranes/trained_models")
        path_model = path_model / "relaynet_model_w_augs_10_21_2020.model"

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
# %%

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

# %%
# calculate layer thickness from labels mask
# generate plots from calculations


# Hide RankWarning for RANSACregression
import warnings
warnings.filterwarnings(action="once")


list_decidua_thickness = []
list_chorion_thickness = []
list_spongy_thickness = []
list_amnion_thickness = []

list_layer_thickness = [list_decidua_thickness,
                        list_chorion_thickness, list_spongy_thickness, list_amnion_thickness]

edges_decidua = []
edges_chorion = []
edges_spongy = []
edges_amnion = []
list_layer_edges = [edges_decidua, edges_chorion, edges_spongy, edges_amnion]


for frame_num, (image, labels) in enumerate(zip(list_images[:10], list_inferences), start=1):
    print(
        f"calculating layer thickness for image: {frame_num}/{len(list_images)}")
    pass

    dict_layers = {
        1: "decidua",
        2: "chorion",
        3: "spongy",
        4: "amnion"
    }

    # iterate through each layer
    # calculate layer thickness
    # exclude top and bottom layers of placenta
    for layer_num, list_data_layers, list_data_edges in zip((np.unique(labels)[1:-1]), list_layer_thickness, list_layer_edges):
        pass
        # convert layer to ints
        layer_mask = (labels == layer_num).astype(int)

        debug = False
        # calculate layer length
        #skel = skeletonize(layer_mask, method='lee').astype(int)

        # CLEAN UP MASKS

        # layer_mask = list_inferences[12] == 1
        if debug:
            plt.title(
                f"original image: {dict_layers[layer_num]} frame {frame_num}")
            plt.imshow(image)
            plt.show()
            plt.title(
                f"original mask: {dict_layers[layer_num]} frame {frame_num}")
            plt.imshow(layer_mask.transpose())
            plt.show()

        # binary close (dialte and erode) to fill holes
        # layers are vertcal so make a vertical rectangular element
        selem_rect = morphology.rectangle(15, 5)
        layer_mask = morphology.binary_closing(
            layer_mask, selem_rect).astype(int)
        if debug:
            plt.title(
                f"after binary closing: {dict_layers[layer_num]} frame {frame_num}")
            plt.imshow(layer_mask.transpose())
            plt.show()

        ### start function for thickness calculation
        layer_thickness, list_poly_coeffs, list_mask_pixels = compute_layer_thickness(layer_mask, method=2, debug=True)


        # add to list for plotting
        list_data_layers.append(layer_thickness)

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
               
        ## make masks to draw over
        
        # list_data_edges.append(mask_edges)

        # # PLOT IMAGE, MASK AND EDGES FOR DEBUGGING
        # # if layer_num == 1:  # which layer to debug
        # plt.title(f"{dict_layers[layer_num]} edges from frame {frame_num}")

        # # format original image
        # plot_image = image[:, 50:-50]
        # plot_image = cv2.cvtColor(plot_image, cv2.COLOR_GRAY2RGB)

        # # add layer shape labels
        # plot_labels = layer_mask
        
        # # take last 265 pixels which are what corresponds to original image
        # # because we added padding on top of the image
        # plot_labels = plot_labels[..., -265:].transpose()
        # plot_labels = plot_labels.astype(np.float32)
        # plot_labels = cv2.cvtColor(plot_labels, cv2.COLOR_GRAY2RGB)
        # blue = (81, 220, 220)
        # l_mask = layer_mask.astype(bool)[..., -265:].transpose()
        # plot_labels[l_mask] = blue
        # plot_labels = plot_labels.astype(np.uint8)
        # overlayed_im_labels = cv2.addWeighted(
        #     plot_image, 1, plot_labels, 0.5, 0)


        # plt.imshow(overlayed_im_labels)
        # plt.show()
        # print(f"frame: {frame_num} top layer length: {layer_top_length}  bottom layer length: {layer_bottom_length}  mask area: {mask_area}  layer thickness: {layer_thickness}")



# %%## combine all edges into a single mask 
edge_masks = []

for pos, (e_decidua, e_chorion, e_spongy, e_amnion) in enumerate(zip(*list_layer_edges)):
    combined_mask = e_decidua + e_chorion + e_spongy + e_amnion
    edge_masks.append(combined_mask)

# save labeled mask
# make tiffs of original, colored overlayed predictions and edges
tiff_stack = np.empty((len(list_inferences), *idx.transpose().shape))
print("saving tiff overlayed output")
out_path = str(path_image.parent / f"{path_image.stem}_with_overlays.tiff")

# out_path = str( f"/home/skalalab/Desktop/{path_image.stem}_combined_edges.tiff")
with tifffile.TiffWriter(out_path, bigtiff=True) as tif:  # imagej=True
    for pos, (image, labels, edges) in enumerate(zip(list_images, list_inferences_colored, edge_masks)):
        pass
        # tiff_stack[pos,...] = labels.transpose().astype(int)
        print(f"saving image {pos+1}/{len(list_images)}")

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
        edges = edges[-265:,...]
        edges[...,0] = 0
        edges[...,2] = 0
        
        overlayed_edges = cv2.addWeighted(image, 1, edges,1, 0)

        # stack images
        # crop edges to original image size (265,3000)
        stack = np.vstack((image, overlayed_im_mask, overlayed_edges)) # why this 247 again?
        # stack = np.vstack((image, overlayed_im_mask))

        # add frame number to output image
        stack = cv2.putText(np.array(
            stack), f"{pos}", (2700, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)
        # plt.imshow(stack)

        tif.save(stack.astype(np.float32))
print(f"finished saving combined image: {out_path}")

# %%
# plot layer thicknesses
for thickness_data in list_layer_thickness:
    plt.plot(thickness_data)
    # plots.append(plt.plot(thickness_data))

plt.xlabel("frame(_/sec)")
plt.ylabel("thickness (pixels)")
plt.title(f"{path_image.stem}")
plt.legend(list(dict_layers.values()))
path_fig_output = path_image.parent / \
    f"{path_image.stem}_layer_thickness_plot.jpeg"
print(f"figure saved in: {str(path_fig_output)}")
plt.savefig(str(path_fig_output))
plt.show()

# plot chorion parameters changing
# dict_layers = {
#     1 : "length top layer",
#     2 : "length bottom layer",
#     3 : "area",
#     4 : "thickness calculation"
# #     }

# # plot top layer:
# plt.plot(chorion_length_top)
# plt.xlabel("frame(_/sec)")
# plt.ylabel("length (pixels)")
# plt.title(f"{path_image.stem}")
# plt.legend(["top layer length"])
# plt.show()

# #bottom layer
# plt.plot(chorion_length_bottom)
# plt.xlabel("frame(_/sec)")
# plt.ylabel("length (pixels)")
# plt.title(f"{path_image.stem}")
# plt.legend(["bottom layer length"])

# plt.show()

# # area
# plt.plot(chorion_area)
# plt.xlabel("frame(_/sec)")
# plt.ylabel("length (pixels)")
# plt.title(f"{path_image.stem}")
# plt.legend(["layer area"])
# plt.show()

# #thickness
# plt.plot(list_chorion)
# plt.xlabel("frame(_/sec)")
# plt.ylabel("length (pixels)")
# plt.title(f"{path_image.stem}")
# plt.legend(["layer thickness"])
# plt.show()

# # for data in [chorion_length_top, chorion_length_bottom, chorion_area, list_chorion]:
