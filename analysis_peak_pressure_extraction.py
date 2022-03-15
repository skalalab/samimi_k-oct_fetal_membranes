from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm

path_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files")

list_path_files = list(path_dataset.glob("*"))
list_path_sample_dirs = [p for p in list_path_files if p.is_dir()]

list_str_path_sample_dirs = [str(p) for p in list_path_sample_dirs]

#%%

dict_peak_pressures = {}

### keys

# (pericervical,periplacental)
# df = pd.DataFrame(columns=["layers", "location", "term","max_pressure"]) # "sample", "subsample",

for path_sample in tqdm(list_path_sample_dirs[:]):
    pass
    # print(path_sample)
    sample = path_sample.stem
    
    path_subsample = path_sample / "Inflation"
    path_all_files_subsample = list(path_subsample.glob("*")) # paths to all subsamples
    list_str_all_files_subsample = [str(p) for p in path_all_files_subsample if p.is_dir()]
    
    # for each subsample
    for subsample_dir in list_str_all_files_subsample:
        pass
   
        # skip newspaper sample in 
        # Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files\2018_10_10_postterm_labor_AROM_41w2d\Inflation
        if "Newspaper" in subsample_dir:
            continue
        
        # build list of all pressure files in subsample 
        list_pressure_files = list(Path(subsample_dir).glob("*ressure*txt"))
        list_early_2020_pressure_files = list(Path(subsample_dir).glob("*Mode2D.txt"))
        list_all_pressure_files = list_pressure_files + list_early_2020_pressure_files
        
        # merge all pressure files into a single df
        df_subsample_pressures = pd.DataFrame(columns=["date", "pressure"])
        for file_pressure in list_all_pressure_files:
            pass
            df_temp = pd.read_csv(file_pressure,delimiter=";", names=["date", "pressure"])
            df_subsample_pressures = pd.concat([df_subsample_pressures, df_temp])
        
        # variables 
        max_pressure = df_subsample_pressures["pressure"].max()
        # print(max_pressure)
        
        subsample_name = Path(subsample_dir).stem
        
        
        # DETERMINE ROW ENTRIES 
        # reset vars
        location = ""
        layers = ""
        pregnancy = ""
        
        # location
        print(subsample_name)
        if "cervical" in subsample_name: location = "pericervical"
        elif "placental" in subsample_name: location = "periplacental"
        else: location = "not found"
        
        # layers
        if "amniochorion" in subsample_name: layers = "amniochorion"
        elif "amnion" in subsample_name: layers = "amnion"
        elif "chorion" in subsample_name: layers = "chorion"
        else: layers = "not found"
        
        # labored/unlabored
        term = "unlabored" if "C_section" in subsample_name else "labored"
        
        # POPULATE DATAFRAME for sample
        # "sample": [sample], "subsample": [subsample_name],
        data = { "location" : [location] , "layers": [layers], "birth_type":term, "max_pressure": [max_pressure]}
        df = pd.DataFrame(data=data)
        # df = df.set_index("sample", drop=True)
        
        df = df.reset_index(drop=True)
        path_sample_output = Path(subsample_dir)
        df.to_csv(path_sample_output / f"{Path(path_sample_output).stem}_max_pressure.csv")

        
        # this saves it all into a single summary df
        # df = df.append(pd.DataFrame(data=data))
        
    # amniochorion and pericervical
    # list_path_amniochorion_pericervical = list(filter(re.compile("pericervical").search, list_path_amniochorion))[0]

# set index to sample and drop index
# df = df.set_index("sample", drop=True)
