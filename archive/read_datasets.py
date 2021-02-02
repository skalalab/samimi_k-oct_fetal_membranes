# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:23:28 2020

@author: May_W
"""
import tifffile
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import glob
import numpy as np

def filesearch(path, word=""):
    """
    Returns a list with all files with the word/extension in it

    Parameters
    ----------
    path : Path object from pathlib
        the folder in which files are searched for.
    word : string
        the key words that are included in the names of the target files. The default is "".
        if searching for a particular extension, enter the string starting with . 
        for example: '.txt'

    Returns a list of full paths
    -------
    file : TYPE
        a list of full paths, which are the directories of the files looked for

    """
    file = []
    for f in glob.glob(str(path / "*")):
        if word[0] == ".":
            if f.endswith(word):
                file.append(f)
        elif word in f:
            file.append(f)
    return file

def extract_image_from_movie(movie_path, folder, index_list = [] ):
    """
    extract images of given indices from the movie. saved to the specified 
    folder. for example: 'image_seq' and 'images'.

    Parameters
    ----------
    movie_path : string
         the folder the movie is in, not include the .tiff file
    folder: string
        the folder where the images will be saved
    index_list : list of ints
        the indices of frames to be extracted and saved

    Returns 
    -------
    None.

    """
    path = filesearch(Path(movie_path), ".tiff")
    path = Path(path[0])
    movie = tifffile.imread(path)
    print(movie.shape)
    if len(index_list) == 0:
  
        roi_index = glob.glob(str(movie_path) + '/roi_files/*')
      
        for f in roi_index:
            index = Path(f).stem
            index = index.split("_")
            i = index[len(index)-1]
            index_list.append(int(i))
        
    for i in index_list:
        img = movie[i,:,:,0]
        file_name = path.stem
        file_name = file_name.split(".")
        # TODO: change the file name
        tifffile.imwrite( f"{movie_path}/{folder}/{file_name[0]}_{i}.tiff", img)
        

def extract_img_from_Img_seq(base_path, keywords): 
    """
    copy the images with corresponding roi files to a folder called 'images' from 'image_seq' folder
    pass in the root directory and the keywords of the folder to be processed.
    Parameters
    ----------
    base_path : string 
        the directory where all the dataset folders are saved. ex. '/image_labels_dataset'
    keywords : list of strings
        keywords for the dataset you want to modify

    Returns
    -------
    None.

    """
    dataset_paths = []
    for kw in keywords:
        dataset_paths += filesearch(Path(base_path), kw)

    for path in dataset_paths:
        filename = Path(path).stem
        for pth in glob.glob(path + '/roi_files/*'):
            p = Path(pth).stem
            index = p.split("_")
            i = index[len(index)-1]
            #shutil.copy(path + '/image_seq/' +filename[4:] + f'_{i}.tiff', path + '/images/' +filename[4:] + f'_{i}.tiff')
            shutil.copy(path + '/image_seq/' +filename + f'_{i}.tiff', path + '/images/' +filename + f'_{i}.tiff')

    
def get_img_roi_dirs(base_path, keywords):
    """
    return the list of full path of rois and images for training

    Parameters
    ----------
    base_path : string
        the folder containing all the dataset folder. ex: '/image_labels_dataset'
    keyword : string
        keyword to identify data folders. ex.'human' or '2018'

    Returns
    -------
    image_list : list
        a list of full paths of all images for training
    roi_list : list
        a list of full paths of all Rois for training

    """
    image_list = []
    roi_list = []
    dataset_paths = []
    for kw in keywords:
        dataset_paths += filesearch(Path(base_path), kw)
    #print(dataset_paths)
    for path in dataset_paths:
        name = Path(path).stem
        for pth in glob.glob(path + '/roi_files/*'):
            roi_list.append(pth)
        
        for p in glob.glob(path + '/images/*'):
            image_list.append(p)
    return image_list, roi_list

def rename_files(base_path, folder_names, is_roi = True):
    """
    rename the roi files to folder name plus frame index

    Parameters
    ----------
    base_path : Path object
        the directory where the dataset folders are saved. ex.'/image_labels_dataset'
    folder_names : list
       list of folder names where target files are stored
    is_roi : boolean, optional
        the folder to be renamed is roi_files. The default is True.
        if False, this method would rename the images

    Returns
    -------
    None.

    """
    for name in folder_names:
        
        if '\\' in name:
            path = str(base_path/Path(name).stem)
            
        else:
            path = str(base_path/name) 
        #print(path)
        if is_roi:
            file_list = glob.glob(path +'/roi_files/*')
           
            for f in file_list:
                index = Path(f).stem
                index2 = index.split("_")
                i = index2[len(index2)-1]
                if '\\' in name:
                    shutil.move(f, path +'/roi_files/' + f'{Path(name).stem}_roi_{i}.zip')
                else:    
                    shutil.move(f, path +'/roi_files/' + f'{name}_roi_{i}.zip')
        else:
            file_list = glob.glob(path +'/image_seq/*')
           
            for f in file_list:
                index = Path(f).stem
                index2 = index.split("_")
                i = index2[len(index2)-1]
                if '\\' in name:
                    shutil.move(f, path +'/image_seq/' + f'{Path(name).stem}_{i}.tiff')
                else:    
                    shutil.move(f, path +'/image_seq/' + f'{name}_{i}.tiff')
            


    
###################################---main---##################################
# =============================================================================
# index = np.arange(0,310,10).tolist()
# lookfor = ['new']
# 
# root = Path('Y:/skala/0-Projects and Experiments/KS - OCT membranes/image_labels_dataset')
# 
# files = []
# for w in lookfor:
#     files = filesearch(root,w)
# 
# for f in files:
#     #extract_image_from_movie(f,'image_seq', index_list = index)
#     print(Path(f).stem)
# rename_files(root, files)
#     
# 
# image_list = []
# roi_list = []
# image_list, roi_list = get_img_roi_dirs(str(root), 'human')
#     
# img = tifffile.imread(image_list[2])
# plt.imshow(img)
# 
# l = ['Kayvan_weka_human_placenta_segmentation']
# rename_files(root,l,is_roi=False)
# extract_image_from_movie(str(root/l[0]),'images', index)
# 
# keyw = ['new']
# a, b = get_img_roi_dirs(str(root), keyw)
# 
# extract_img_from_Img_seq(str(root), l)
# 
# =============================================================================
# =============================================================================
# from pathlib import Path
# root = Path('Y:/skala/0-Projects and Experiments/KS - OCT membranes/image_labels_dataset')
# index = np.arange(0,321,10).tolist()
# lookfor = ['2019_03_06_human_amniochorion_labored_term_AROM_periplacental_0002_Mode2D',
#           '2018_12_12_human_amniochorion_labored_term_SROM_pericervical_0002_Mode2D',
#           '2018_12_12_human_amniochorion_labored_term_SROM_periplacental_0002_Mode2D',
#           '2018_10_09_human_amniochorion_labored_term_AROM_periplacental_0002_Mode2D']
# files =[]
# for w in lookfor:
#      files += filesearch(root,w)
# for f in files:
#     i = []
#     for pth in glob.glob(f + '/roi_files/*'):
#             p = Path(pth).stem
#             index = p.split("_")
#             i.append( int(index[len(index)-1]))
#        # i = np.arange(0,501,10).tolist()
#     extract_image_from_movie(f, index_list = i)
#    
#     print(Path(f).stem)
# =============================================================================
     
     
     
     
     
     
     
     
     