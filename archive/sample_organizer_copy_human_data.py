#%% This script copies over files from Kayvan's Human Data folder while keeping directory structure, 
# files already copied are skipped

from pathlib import Path
import shutil

path_complete_dataset = Path(r"Z:\Kayvan\Human Data".replace("\\",'/'))
list_all_files = list(path_complete_dataset.glob("*"))
list_samples = [folder for folder in list_all_files if folder.is_dir()]

# This path will store copy of files and structure
path_output = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files".replace("\\",'/'))

list_no_samples_found = []
list_no_oct_subsamples = []

for num_sample, sample in enumerate(list_samples, start=1) :
    pass
    print(f"{num_sample}/{len(list_samples)}  |  {sample.name}")
    
    path_inflation = sample / "Inflation"
       
    list_subsamples = list(path_inflation.glob("*"))
    
    if len(list_subsamples) == 0:
        print("no samples found")
        list_no_samples_found.append(sample)
    
    for num_subsample, subsample in enumerate(list_subsamples, start=1):
        pass
        print(f"{num_subsample}/{len(list_subsamples)}  |  {subsample.name}")
        
        # mkdir subsample dir and parents
        path_output_subsample = path_output / sample.name /"Inflation" / subsample.name
        path_output_subsample.mkdir(parents=True, exist_ok=True)
    
        ## copy oct files 
        list_path_subsample_oct_tiffs = list(subsample.glob("*Mode2D.tiff"))  
        if len(list_path_subsample_oct_tiffs) == 0:
            print("no oct samples found")
            list_no_oct_subsamples.append(subsample)
    
        for path_file in list_path_subsample_oct_tiffs:
            pass
            if not (path_output_subsample /  path_file.name).exists():
                    print(f"copying: {path_file}")
                    # shutil.copy2(path_file, path_output_subsample)
        
        # copy txt files 
        list_path_subsample_pressure_files = list(subsample.glob("*.txt"))
        for path_file in list_path_subsample_pressure_files:
            pass
            if not (path_output_subsample /  path_file.name).exists():
                print(f"copying: {path_file}")
                # shutil.copy2(path_file, path_output_subsample)

    
    
#%%
# import re


# ## processed samples
# path_segmented_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\oct_dataset_3100x256\0-segmentation_completed".replace("\\",'/'))
# list_path_processed = list(path_segmented_dataset.glob("*processed*"))

# # find tiff filename of samples to move
# # for each filename, split at first dash and keep first part
# list_path_dir_names = [str(p.name).rsplit("-",1)[0] for p in list_path_processed]



# # find corresponding folders 
# path_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files".replace("\\",'/'))

# # list directories
# list_path_all_files = list(path_dataset.rglob("*")) # get all files in this directory
# list_str_path_all_files = [str(p) for p in list_path_all_files]

# for procesed_sample in list_path_dir_names:
   
#     path_corresponding_dir = list(filter(re.compile(procesed_sample).search, list_str_path_all_files ))
#     print(f"{procesed_sample }==> {Path(path_corresponding_dir[0]).parent.parent.parent.name}")
    
    
    
    
    
    
    
    
    
    
    