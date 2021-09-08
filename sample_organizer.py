

from pathlib import Path
from pprint import pprint 
# import shutil

path_complete_dataset = Path(r"Z:\Kayvan\Human Data".replace("\\",'/'))
list_all_files = list(path_complete_dataset.glob("*"))
list_samples = [folder for folder in list_all_files if folder.is_dir()]

# This path will store copy of files and structure
path_output = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files".replace("\\",'/'))

list_no_samples_found = []
list_no_oct_subsamples = []

for num_sample, sample in enumerate(list_samples[0:1], start=1) :
    print(f"{num_sample}/{len(list_samples)}  |  {sample.name}")
    
    path_inflation = sample / "Inflation"
    
    path_output_inflation = path_output / "Inflation"
    
    ## Make dir
    # path_output_inflation.mkdir(parents=True)
    
    list_subsamples = list(path_inflation.glob("*"))
    
    if len(list_subsamples) == 0:
        print("no samples found")
        list_no_samples_found.append(sample)
        
    
    for subsample in list_subsamples:
        pass
        
        # mkdir subsample
        path_output_subsample = path_output_inflation / subsample.name
        # path_output_subsample.mkdir()
    
        ## copy oct files 
        list_path_subsample_oct_tiffs = list(subsample.glob("*Mode2D.tiff"))
        
         
        if len(list_path_subsample_oct_tiffs) == 0:
            print("no oct samples found")
            list_no_oct_subsamples.append(subsample)
    
        for file in list_path_subsample_oct_tiffs:
            pass
            src = file
            dst = path_output_subsample / file.name
            #shutil.copy2()
        
        # copy txt files 
        list_path_subsample_pressure_files = list(subsample.glob("*.txt"))
        for file in list_path_subsample_pressure_files:
            pass
            src = file
            dst = path_output_subsample / file.name
            # shutil.copy2()

        ####
        
        print(f"{'-'*10} {sample.name}" )
        print("oct files:")
        pprint(list_path_subsample_oct_tiffs)
        print("pressure files:")
        pprint(list_path_subsample_pressure_files)
        
        #copy files --> check if they are already in the folder
        #  make a dir for each sample
        # from each dir, copy 
    
    
    
    
    