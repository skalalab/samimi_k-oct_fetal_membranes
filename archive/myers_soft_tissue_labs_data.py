# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:52:03 2021

@author: Nabiki
"""

from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent

# path_data = Path("Z:/0-Projects and Experiments/KS - OCT membranes/ms")
# list_path_samples = list(path_data.glob("*[0-9]d*"))

# path_output = HERE / "data_for_soft_tissue_lab"
# path_output = Path("C:/Users/Nabiki/Desktop/soft_tissue_lab")
path_output = Path("F:/Emmanuel/pressure_data")



####
# list_sample = Path(r"Z:\Kayvan\Human Data\2018_12_12_term_labor_SROM_39w5d".replace("\\","/"))
list_sample = Path(r"Z:\Kayvan\Human Data\2018_10_09_term_labor_AROM_39w5d")

list_path_samples = [list_sample]

####

for path_sample in list_path_samples:
    print(f"{'-'*15}sample: {str(path_sample.name)}{'-'*15}")
    
    # MAKE DIR FOR CURRENT SAMPLE
    path_output_sample_dir = path_output / path_sample.name
    path_output_sample_dir.mkdir(exist_ok=True)
    
    # 2P
    print(f"==> 2p dir")
    path_2p_dir = path_sample / "2p"
    
    
    # INFLATION
    print(f"==> inflation dir")
    # get all subdirs
    path_inflation_dir = path_sample / "inflation"
    list_path_inflation_samples = [p for p in list(path_inflation_dir.glob("*")) if p.is_dir()]
    

    # create output inflation dir
    path_output_inflation = path_output_sample_dir / "inflation"
    path_output_inflation.mkdir(exist_ok=True)
    
    #iterate through the dirs
    for path_inf_sample in list_path_inflation_samples:
        pass
    
        # make output dir
        path_output_inflation_subsample = path_output_inflation / path_inf_sample.name
        path_output_inflation_subsample.mkdir(exist_ok=True)
        #print(f"output dir: {str(path_output_inflation_subsample.name)}")
        
        # pressure.txt
        list_pressure_files = list(path_inf_sample.glob("*_?ressure.txt"))
        for path_file in list_pressure_files:
            pass
            # copying files
            print(f"copying: {path_file.name} ")
            # copy files
            shutil.copy2(path_file, path_output_inflation_subsample)
        
        
        # thickness_values.csv
        list_thickness_files = list(path_inf_sample.glob("*thickness_values.csv"))

        for path_file in list_thickness_files:
            pass
            # copying files
            print(f"copying: {path_file.name} ")
            # copy files
            shutil.copy2(path_file, path_output_inflation_subsample)
            
            
        # VideoImage.tiff
        list_VideoImage = list(path_inf_sample.glob("*_VideoImage.tiff"))
        for path_file in list_VideoImage:
            pass
            # copying files
            print(f"copying: {path_file.name} ")
            # copy files
            shutil.copy2(path_file, path_output_inflation_subsample)
            
        # Pressure_Analysis.mat
        list_pressure_analysis = list(path_inf_sample.glob("*_Pressure_Analysis.mat"))
        for path_file in list_pressure_analysis:
            pass
            # copying files
            print(f"copying: {path_file.name} ")
            # copy files
            shutil.copy2(path_file, path_output_inflation_subsample)
            
        # Pressure_Stress_Strain.tif
        list_pressure_stress_strain_analysis = list(path_inf_sample.glob("*_Pressure_Stress_Strain.tif"))
        for path_file in list_pressure_stress_strain_analysis:
            pass
            # copying files
            print(f"copying: {path_file.name} ")
            # copy files
            shutil.copy2(path_file, path_output_inflation_subsample) 
            
              
    
    # SUTURE
    print(f"==> suture dir")
    path_suture_dir = path_sample / "Suture" # path to sample suture dir
    
    # make output suture dir
    path_output_suture = path_output_sample_dir / "Suture"
    path_output_suture.mkdir(exist_ok=True)
    
    # xlsx files
    list_path_xlsx_files = list(path_suture_dir.glob("*.xlsx"))
    for path_file in list_path_xlsx_files:
        print(f"copying: {path_file.name}" )
        
        # copy files
        shutil.copy2(path_file, path_output_suture)
        
    # bmp files
    list_path_bmp_files = list(path_suture_dir.glob("*.bmp"))
    for path_file in list_path_bmp_files:
        print(f"copying: {path_file.name}" )
        
        # copy files
        shutil.copy2(path_file, path_output_suture)
           
    
    
    














