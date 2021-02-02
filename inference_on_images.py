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


# directory to hold split masks
path_split_masks = path_image.parent / "split_masks"
path_split_masks.mkdir(exist_ok=True)

for frame_num, (image, labels) in enumerate(zip(list_images[:], list_inferences[:]), start=1):
    print(
        f"calculating layer thickness for image: {frame_num}/{len(list_images)}")
    pass

    # export images for Dan
    temp_labels = labels.transpose()
    path_image_output = path_split_masks / \
        f"{path_image.stem}_image_{str(frame_num).zfill(4)}.tiff"
    tifffile.imsave(path_image_output, image)
    for mask_value, layer_name in zip(np.unique(labels)[1:-1], ["decidua", "chorion", "spongy", "amnion"]):
        pass
        path_mask_output = path_split_masks / \
            f"{path_image.stem}_{layer_name}_{str(frame_num).zfill(4)}.tiff"
        single_mask = (labels == mask_value).astype(bool).transpose()
        # tifffile.imsave(path_mask_output, single_mask)
        plt.title(f"{layer_name} frame {frame_num}")
        plt.imshow(single_mask)
        plt.show()
    # END

    dict_layers = {
        1: "decidua",
        2: "chorion",
        3: "spongy",
        4: "amnion"
    }

    # calculate layer thickness
    # exclude top and bottom layers of placenta
    for layer_num, list_data_layers, list_data_edges in zip((np.unique(labels)[1:-1]), list_layer_thickness, list_layer_edges):

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

        # KEEP LARGEST CONNETED COMPONENT
        # this enumerates blobs ! not binary mask anymore!
        label_image = label(layer_mask)

        label_region_props = regionprops(label_image)
        for pos, region in enumerate(label_region_props):
            # print(f"pos: {pos} area: {region.area}")
            if pos == 0:   # initialize with first region
                largest_region = region  # initialize largest region
            elif region.area > largest_region.area:
                largest_region = region

        # KEEP LARGEST REGION OR REMOVE ALL LARGER AREAS
        # remove everything smaller than largest mask
        # layer_mask = remove_small_objects(label_image, min_size=largest_region.area-1) # keek largest region
        layer_mask = remove_small_objects(label_image, min_size=(
            largest_region.area*0.1))  # remove spots < 10% largest region found

        # make labels mask binary, otherwise area will be off
        layer_mask[layer_mask > 0] = 1

        # alternatively can keep largest region to use, significantly speeds up processing
        # layer_mask = largest_region.image

        # if debug:
        # if layer_num == 2:
        #     plt.title(f"largest region: {dict_layers[layer_num]} frame {frame_num}")
        #     plt.imshow(layer_mask.transpose())
        #     plt.show()
        #     break

        # calculate layer length 2nd method
        # placenta is  vertical here, meaning edges are at the top and bottom
        # layers increase left to right
        def _find_layer_vertices(mask):
            list_layer_vertices = []
            for pos_row, row in enumerate(mask):  # iterate through rows
                # iterate through cols in row
                for pos_col, pixel_value in enumerate(row):

                    if (pos_col+1) == len(row):  # check if we've reached the end of the array
                        break  # goto next row

                    if pixel_value == True:  # mask is at edge of mask
                        list_layer_vertices.append(
                            (pos_row, pos_col))  # save layer pixel
                        break  # go to next row

                    # next pixel is a layer pixel if different
                    if pixel_value != row[pos_col+1]:
                        list_layer_vertices.append(
                            (pos_row, pos_col+1))  # save layer pixel
                        break  # go to next row
            return list_layer_vertices

        # generate edge mask from vertices
        def _generate_layer_edge_mask(list_vertices, mask_shape, show_image=False):
            mask_layer_edge = np.zeros(mask_shape)  # mask_array
            for pos, vertex in enumerate(list_vertices):
                if (pos+1) == len(list_vertices):  # reached last vertex, exit
                    break
                curr_pixel_row, curr_pixel_col = vertex
                next_row_pixel, next_col_pixel = list(
                    list_vertices[pos+1])  # get next vertex
                line_pixels = line(curr_pixel_row, curr_pixel_col,
                                   next_row_pixel, next_col_pixel)

                # fill in pixels in array
                for pos, (pixel_row, pixel_col) in enumerate(zip(*line_pixels)):
                    # fill edges with ones
                    mask_layer_edge[pixel_row, pixel_col] = 1

            if show_image:
                plt.imshow(mask_layer_edge, vmin=0, vmax=1)
                plt.show()

            return mask_layer_edge

        # calculate top edge
        list_top_pixels = _find_layer_vertices(layer_mask)
        mask_top_edge = _generate_layer_edge_mask(
            list_top_pixels, layer_mask.shape, show_image=False)
        if debug:
            plt.title(
                f"{dict_layers[layer_num]} top edge from frame {frame_num}")
            plt.imshow(mask_top_edge.transpose())
            plt.show()

        # calculate bottom layer thickness
        list_bottom_pixels = _find_layer_vertices(np.flip(layer_mask, axis=1))
        mask_bottom_edge = _generate_layer_edge_mask(
            list_bottom_pixels, layer_mask.shape, show_image=False)
        if debug:
            plt.title(
                f"{dict_layers[layer_num]} bottom edge from frame {frame_num}")
            plt.imshow(np.flipud(mask_bottom_edge.transpose()))
            plt.show()

        layer_top_length = np.sum(mask_top_edge)
        layer_bottom_length = np.sum(mask_bottom_edge)
        average_layer_thickness = (layer_top_length + layer_bottom_length)/2

        mask_area = np.sum(layer_mask)
        layer_thickness = np.sum(layer_mask)/average_layer_thickness

        # add to list for plotting
        list_data_layers.append(layer_thickness)

        # save edges to plot later
        mask_edges = (mask_top_edge + np.flip(mask_bottom_edge)).transpose()
        list_data_edges.append(mask_edges)

        # debug edges
        if layer_num == 1:  # which layer to debug
            plt.title(f"{dict_layers[layer_num]} edges from frame {frame_num}")

            # format original image
            plot_image = image[:, 50:-50]
            plot_image = cv2.cvtColor(plot_image, cv2.COLOR_GRAY2RGB)

            # add layer shape labels
            plot_labels = layer_mask
            # take last 265 pixels
            plot_labels = plot_labels[..., -265:].transpose()
            plot_labels = plot_labels.astype(np.float32)
            plot_labels = cv2.cvtColor(plot_labels, cv2.COLOR_GRAY2RGB)
            blue = (81, 220, 220)
            l_mask = layer_mask.astype(bool)[..., -265:].transpose()
            plot_labels[l_mask] = blue
            plot_labels = plot_labels.astype(np.uint8)
            overlayed_im_labels = cv2.addWeighted(
                plot_image, 1, plot_labels, 0.5, 0)

            # color edges in purple
            # keep last 265 rows to match original image
            plot_mask_edges = mask_edges[-265:, :]
            e_mask = plot_mask_edges  # for coloring purposes
            plot_mask_edges = plot_mask_edges.astype(np.float32)
            plot_mask_edges = cv2.cvtColor(plot_mask_edges, cv2.COLOR_GRAY2RGB)
            plot_mask_edges = plot_mask_edges.astype(np.uint8)
            edge_color = (0, 255, 0)

            plot_mask_edges[e_mask.astype(bool)] = edge_color
            # combine images
            overlayed_im_edges = cv2.addWeighted(
                overlayed_im_labels, 1, plot_mask_edges, 1, 0)

            plt.imshow(overlayed_im_edges)
            plt.show()
            print(f"frame: {frame_num} top layer length: {layer_top_length}  bottom layer length: {layer_bottom_length}  mask area: {mask_area}  layer thickness: {layer_thickness}")

        # for troubleshooting, look at chorion first
        # if layer_num == 1: # this is the chorion
        #     chorion_length_top.append(layer_top_length)
        #     chorion_length_bottom.append(layer_bottom_length)
        #     chorion_area.append(mask_area)
        #     # chorion_edges = (mask_top_edge + np.fliplr(mask_bottom_edge)).transpose()
        #     # plt.imshow(chorion_edges[:100,600:800])
        #     # plt.show()
        #     # list_chorion_edges.append(chorion_edges)


# %%## combine edges and save tiff, only if original dimensions kept across all masks (line 262)
edge_masks = []
# imagej=True
with tifffile.TiffWriter(f"/home/skalalab/Desktop/{path_image.stem}_edges.tiff", bigtiff=True) as tif:
    # iterate through edge lists
    for pos, (e_decidua, e_chorion, e_spongy, e_amnion) in enumerate(zip(*list_layer_edges)):
        combined_mask = e_decidua + e_chorion + e_spongy + e_amnion
        edge_masks.append(combined_mask)
        # tif.save(combined_mask.astype(np.float32))
# %%

# save labeled mask
# make tiffs of original, colored overlayed predictions and edges
tiff_stack = np.empty((len(list_inferences), *idx.transpose().shape))
print("saving tiff combined output")
out_path = str(path_image.parent / f"{path_image.stem}_combined_edges.tiff")
with tifffile.TiffWriter(out_path, bigtiff=True) as tif:  # imagej=True
    for pos, (image, labels, edges) in enumerate(zip(list_images, list_inferences_colored, edge_masks)):
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
        labels = labels[-265:, ...]  # take last 265 pixels

        overlayed_im_mask = cv2.addWeighted(image, 1, labels, 0.5, 0)
        # plt.imshow(overlayed_im_mask)

        # add edges
        edges = edges.astype(np.float32)
        edges[edges > 0] = 255
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # stack images
        # crop edges to original image size (265,3000)
        stack = np.vstack((image, overlayed_im_mask, edges[247:, :]))
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
    f"{path_image.stem}_plot_largest_region.jpeg"
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
