# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:53:16 2021

@author: OCT
"""

from pathlib import Path
import re
from PIL import Image
import shutil

# GET PATHS TO ALL 65 SAMPLES AND ALL FILES
path_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files")
list_path_all_samples = list(path_dataset.glob("*"))
list_path_all_files = list(path_dataset.rglob("*"))
list_path_str_all_files = [str(p) for p in list_path_all_files]

## for Soft Tissue Lab
path_dataset_soft_tissue_lab = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\dataset_for_soft_tissue_lab")

count_processed_thickness = 0
count_processed_matlab = 0
count_processed_gif = 0
count_processed_features_csv = 0
count_processed = 0
count_total = 0
############## PRINT REPORT
for num_sample, path_sample in enumerate(list_path_all_samples[:], start=1): #modify to display less 
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
    
    # list of gif files
    list_path_str_gif = list(filter(re.compile(".*.gif").search,list_path_str_subsample_all_files ))
    list_gif_filenames_processed = [Path(p).stem for p in list_path_str_gif]
    
    # #list notes
    # list_path_str_notes = list(filter(re.compile(".*Notes.txt").search, list_path_str_subsample_all_files ))
    # list_path_notes = [Path(p).stem for p in list_path_str_notes]
    # if (len(list_path_notes)) !=0:
    #     print(list_path_notes)
    
    # list of gif files
    list_path_str_features = list(filter(re.compile(".*features.csv").search,list_path_str_subsample_all_files ))
    list_features_filenames_processed = [Path(p).stem.rsplit("_",1)[0] for p in list_path_str_features]
    ##################
    
    
    # PRINT REPORT
    for stem_subsample in list_images_to_process:
        pass
        bool_thickness = False
        bool_matlab = False
        bool_gif = False
        bool_features_csv = False
        count_total += 1 # count total images to process
        
        # thickness processed?
        if stem_subsample in list_thickness_filenames_processed:
            bool_thickness = True
            count_processed_thickness += 1
        
        #matlab processed?
        if stem_subsample in list_apex_filenames_processed:
            bool_matlab = True
            count_processed_matlab += 1
            
        # gif included?
        if stem_subsample in list_gif_filenames_processed:
            bool_gif = True
            count_processed_gif +=1
        
        # features
        if stem_subsample in list_features_filenames_processed:
            bool_features_csv = True
            count_processed_features_csv +=1
            
    
        # PRINT REPORT FOR SAMPLE
        if bool_thickness and bool_matlab and bool_gif and bool_features_csv:  
            # samples here are complete
            print(f"{' ' * 8}[x] {stem_subsample}")
            count_processed += 1
            
            # COPY IMAGES TO SOFT TISSUE LAB FOLDER
            # path_soft_tissue_output = path_dataset_soft_tissue_lab / path_sample.name /"Inflation" / stem_subsample
            # if not path_soft_tissue_output.exists():
            #     path_soft_tissue_output.mkdir(parents=True)
            
            # # copy necessary files if they don't already exist
            # # features 
            # path_features_csv = list(filter(re.compile(f".*{stem_subsample}.*").search, list_path_str_features))[0]
            # if not (path_soft_tissue_output / Path(path_features_csv).name).exists():
            #     pass
            #     shutil.copy2(path_features_csv, path_soft_tissue_output)
            
            # # gif
            # path_gif = list(filter(re.compile(f".*{stem_subsample}.*").search, list_path_str_gif))[0]
            # if not (path_soft_tissue_output / Path(path_gif).name).exists():
            #     pass
            #     shutil.copy2(path_gif, path_soft_tissue_output)

            # # notes - copy if it exists for the sample
            # path_notes = (Path(path_gif).parent / "Notes.txt")
            # if not (path_soft_tissue_output / "Notes.txt").exists() and path_notes.exists():
            #         print(path_notes)
            #         shutil.copy2(path_notes, path_soft_tissue_output)

            # assert Path(path_features_csv).exists() and Path(path_gif).exists(), "Error either gif or features_csv path is invalid"
        
        else:
            print(f"{' ' * 8}[ ] {stem_subsample}")
            if not bool_thickness:
                print(f"{' ' * 12}[ ] thickness (ReLayNet)")
            if not bool_matlab:
                print(f"{' ' * 12}[ ] apex rise (matlab script)")
            if not bool_gif:
                print(f"{' ' * 12}[ ] gif")
            if not bool_features_csv:
                print(f"{' ' * 12}[ ] features_csv")
                
            
        
    print("") # print new line if list not empty


        
print('-'*10)
print(f"Total processed: {count_processed}")
print(f"Thickness processed: {count_processed_thickness}/{count_total}")
print(f"Apex rise processed : {count_processed_matlab}/{count_total}")
print(f"gifs processed : {count_processed_gif}/{count_total}")
print(f"features csv : {count_processed_features_csv}/{count_total}")


#%%