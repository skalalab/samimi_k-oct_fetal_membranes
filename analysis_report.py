# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:53:16 2021

@author: OCT
"""

from pathlib import Path
import re
from PIL import Image

# GET PATHS TO ALL 65 SAMPLES AND ALL FILES
path_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files".replace("\\",'/'))
list_path_all_samples = list(path_dataset.glob("*"))
list_path_all_files = list(path_dataset.rglob("*"))
list_path_str_all_files = [str(p) for p in list_path_all_files]

count_processed_thickness = 0
count_processed_matlab = 0
count_processed = 0
count_total = 0
############## PRINT REPORT
for num_sample, path_sample in enumerate(list_path_all_samples[:], start=1):
    print(f"{num_sample})  {path_sample.name}")
    
    ##################
    # ALL FILES IN SUBSAMPLE 
    list_path_subsample_all_files = list(path_sample.rglob("*"))
    list_path_str_subsample_all_files = [str(p) for p in list_path_subsample_all_files]
    
    # FIND POTENTIAL IMAGES TO PROCESS    
    path_subsample_pressure_files= list(path_sample.rglob("*amniochorion*_Pressure.txt"))
    list_path_images = [p.parent / f"{(p.stem.rsplit('_', 1)[0])}.tiff" for p in path_subsample_pressure_files]

    # VALIDATE IMAGES FOUND
    # image exists
    # size: 3100x265
    # not from first half of 2020 (<May)
    list_images_to_process = []
    for path_image in list_path_images:
        pass
        assert path_image.exists(), "Image does not exist: {path_image}"
        im = Image.open(path_image)
        if not im.size == (3100, 265): # make sure images are same size as trained
            print(f"Size Error: {path_image.name} | {im.size}")
            continue
        list_images_to_process.append(path_image.stem)
    
    # LIST PROCESSED IMAGES
    list_path_str_processed = list(filter(re.compile(".*_thickness.csv").search, list_path_str_subsample_all_files))
    list_thickness_filenames_processed = [Path(p).stem.rsplit("_", 1)[0] for p in list_path_str_processed]
    
    
    # List MATLAB Processed
    list_path_str_matlab = list(filter(re.compile(".*_Pressure_Apex.csv").search, list_path_str_subsample_all_files))
    list_apex_filenames_processed = [Path(p).stem.rsplit("_",2)[0] for p in list_path_str_matlab]
    ##################
    
    
    # LIST IMAGES NOT YET PROCESSED
    for stem_subsample in list_images_to_process:
        pass
        bool_thickness = False
        bool_matlab = False
        count_total += 1 # count image to process
        
        # thickness processed?
        if stem_subsample in list_thickness_filenames_processed:
            bool_thickness = True
            count_processed_thickness += 1
        
        #matlab processed?
        if stem_subsample in list_apex_filenames_processed:
            bool_matlab = True
            count_processed_matlab += 1
    
        # PRINT REPORT FOR SAMPLE
        if bool_thickness and bool_matlab:    
            print(f"{' ' * 8}[x] {stem_subsample}")
            count_processed += 1
        else:
            print(f"{' ' * 8}[ ] {stem_subsample}")
            if not bool_thickness:
                print(f"{' ' * 12}[ ] thickness (ReLayNet)")
            if not bool_matlab:
                print(f"{' ' * 12}[ ] apex rise (matlab script)")
                
            
        
    print("") # print new line if list not empty


        
print('-'*10)
print(f"Total processed: {count_processed}")
print(f"Thickness processed: {count_processed_thickness}/{count_total}")
print(f"Apex rise processed : {count_processed_matlab}/{count_total}")
