

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
    