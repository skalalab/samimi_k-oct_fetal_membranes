# -*- coding: utf-8 -*-
from read_roi import read_roi_zip
import os
import tifffile
from matplotlib import pyplot as plt
import numpy as np
import re
from sdtfile import SdtFile
import numpy
from skimage.draw import polygon2mask
import zipfile

''' helper functions '''
def image_show(image, nrows=1, ncols=1): # , cmap='gray'
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    ax.imshow(image) # , cmap='gray'
    ax.axis('off')
    return fig, ax

# given a list of ROIs returns a mask
def create_mask_from_rois(rois_vert):
    width = 256
    full_image_mask = np.zeros((width, width), dtype=np.uint8)
    rois_in_image = []
    rois_vertices = []
    for roi in list(rois_vert.keys()):
        rois_vertices.append(roi)
        col_coords = rois_vert[roi]['x']
        row_coords = rois_vert[roi]['y']
        polygon = [(row_coords[i], col_coords[i]) for i in range(0, len(row_coords))] # create list of values
        #img = Image.new('L', (width, width), 0)
        #ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        #single_roi = numpy.array(img)
        image_shape = (width, width)
        single_roi_mask = polygon2mask(image_shape, polygon)
        full_image_mask = full_image_mask + single_roi_mask # add roi to whole image mask
        rois_in_image.append(single_roi_mask)
        #    plt.imshow(mask)
    binary_mask = full_image_mask > 0
    # image_show(mask)
    # image_show(binary_mask)
    return binary_mask, rois_in_image

def threshold_masks(images, masks, rois_pixel_count):

    refined_roi_masks = []
    thresholds = []

    # mask photon images
    masked_images = []
    for pos, image in enumerate(images):
        masked_images.append(image * masks[pos])

    # iterate through all the images
    for img_idx, masked_image in enumerate(masked_images):
        # threshold from 0 to max num pixels until we get ~ same ammount
        for i in np.arange(np.max(masked_image)):
            thresholded_image = masked_image > i # boolean mask
            if thresholded_image.sum() <= rois_pixel_count[img_idx]:  # count pixels
                print(f'threshold: {i} ')
                refined_roi_masks.append(thresholded_image)
                thresholds.append(i)
                break

    return thresholds, refined_roi_masks

def load_2d_photon_intensity_nadph_tiffs(path, all_files):
    temp_images = []
    r = re.compile('.*n_photons.tiff')
    image_tiffs = list(filter(r.match, all_files)) # zip file rois
    image_tiffs.sort()
    for image_filename in image_tiffs:
        img = tifffile.imread(path + os.sep + image_filename)
        temp_images.append(img)
        #image_show(img)
    return temp_images


def load_roi_masks_from_zip(path):
    # # get list of zip files
    # r = re.compile('.*.zip')
    # roi_zip_filenames = list(filter(r.match, all_files)) # zip file rois
    # roi_zip_filenames.sort()

    #arrays to hold roi sets and masks
    masks = []
    list_roi_sets = []

    # iterate through each file, load rois and make into masks

    # load single zip file
    rois_vertices = read_roi_zip(path)
    mask, rois = create_mask_from_rois(rois_vertices)
    masks.append(mask)
    list_roi_sets.append(rois)
    #image_show(mask)
    return masks, list_roi_sets

def load_tif_masks(directory, regex=''):
    r = re.compile(regex)
    dir_name = ''
    regex = f'.*\.tif' if regex == '' else regex
    r = re.compile(regex)
    all_files_base_dir = os.listdir(directory)
    sdt_filenames = list(filter(r.match, all_files_base_dir))  # zip file rois
    sdt_filenames.sort()
    print(sdt_filenames)

    masks = []
    for file in sdt_filenames:
        mask = tifffile.imread(directory + os.sep + file)
        masks.append(mask)

    # create sets out of masks
    roi_sets = []

    for mask in masks:  # iterate through each images masks
        n_rois = np.max(mask)
        temp_roi_set = []
        for idx in np.arange(1, n_rois):  # iterate through rois
            roi = mask == idx
            temp_roi_set.append(roi)
        roi_sets.append(temp_roi_set)

    return roi_sets

def load_sdt_files(dir_name, directory, regex=''):
    ''' load sdt files ''' #### capture decays
    regex = f'.*{dir_name}_[0-9]n\.sdt' if regex == '' else regex
    # load zip files
    all_files_base_dir = os.listdir(directory)
    width = 256
    timebins = 256
    sdt_images = []
    r = re.compile(regex)
    sdt_filenames = list(filter(r.match, all_files_base_dir))  # zip file rois
    sdt_filenames.sort()
    print(f'sdt_filenames: {sdt_filenames}')

    for sdt_filename in sdt_filenames:
        filepath = directory + os.sep + sdt_filename
        with zipfile.ZipFile(filepath) as myzip:
            z1 = myzip.infolist()[0]  # "data_block" or sdt bruker uses "data_block001" for multi-sdt"
            with myzip.open(z1.filename) as myfile:
                data = myfile.read()
                data = np.frombuffer(data, np.uint16)

        ####
        # sdt = SdtFile(filepath)
        # print(f'data: {data}')
        # print(f'{data.shape}')
        # if len(sdt.data[0]) == 16777216:  # (256,256,256)
        #     sdt_np = np.reshape(sdt.data[0], (width, width, timebins))
        #     sdt_images.append(sdt_np)
        #####
        if len(data) == 33554432:  # (2, 256,256,256)
            sdt_np = np.reshape(data, (2, width, width, timebins))
            sdt_images.append(sdt_np)
        # img = sdt_np[0, :, :, :]
        # plt.imshow(np.sum(img, axis=2)) ## x,y,t
        # plt.show()
    return sdt_images

def load_nadph_tau_m_masks(path, all_files):
    list_filenames_tau_m = list(filter(re.compile('.*NADHtm.tif').match, all_files))
    list_filenames_tau_m.sort()
    list_tau_m_masks = []
    for pos, filename in enumerate(list_filenames_tau_m):
        # load tau_m file and make mask out of it
        nadph_img = tifffile.imread(path + os.sep + filename)
        nadph_img = np.nan_to_num(nadph_img) # nan to num
        nadph_img[nadph_img > 0] = 1 # make binary
        list_tau_m_masks.append(nadph_img)
#        image_show(nadph_img * images[pos])
    return list_tau_m_masks
    # convert to binary
    # multiplly by ROI's and masks

def refined_roi_sets(roi_sets, masks):
    new_roi_sets = []
    for pos, roi_set in enumerate(roi_sets): #iterate through roi sets
        mask = masks[pos]
        image_rois = [] # store each images ROIs
        for roi in roi_set:
            image_rois.append(mask * roi)
        new_roi_sets.append(image_rois)
    return new_roi_sets

#            ## multiply nadph masks with each roi to update them
#''' iterate through the ROIs and update their area'''
#updated_roi_sets = []
#for pos, roi_set in enumerate(roi_sets): # iterate through sets
#    new_set = []
#    for roi in roi_set: # iterate through rois
#        new_set.append(roi * refined_roi_masks[pos])
#    updated_roi_sets.append(new_set)
#