

from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm

path_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files")

list_path_files = list(path_dataset.glob("*"))
list_path_sample_dirs = [p for p in list_path_files if p.is_dir()]

list_str_path_sample_dirs = [str(p) for p in list_path_sample_dirs]

# #split into labored/unlabored(c section)
# list_labored = []
# list_unlabored = []

# for path_str_subsample in list_str_path_sample_dirs:
#     pass
#     if "_C_section" in path_str_subsample:
#         list_unlabored.append(path_str_subsample)
#     else:
#         list_labored.append(path_str_subsample)

#%%

dict_peak_pressures = {}

### keys

# (pericervical,periplacental)
df = pd.DataFrame(columns=["sample", "subsample", "layers", "location", "pregnancy","max_pressure"])

for path_sample in tqdm(list_path_sample_dirs):
    pass
    
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
        
        df_subsample_pressures = pd.DataFrame(columns=["date", "pressure"])
        for file_pressure in list_all_pressure_files:
            pass
            df_temp = pd.read_csv(file_pressure,delimiter=";", names=["date", "pressure"])
            df_subsample_pressures = df_subsample_pressures.append(df_temp)
        
        # variables 
        max_pressure = df_subsample_pressures["pressure"].max()
        # print(max_pressure)
        
        subsample_name = Path(subsample_dir).stem
        
        
        # DETERMINE ROW ENTRIES 
        # location
        if "pericervical" in subsample_name: location = "pericervical"
        elif "periplacental" in subsample_name: location = "periplacental"
        
        # layers
        if "amniochorion" in subsample_name: layers = "amniochorion"
        elif "amnion" in subsample_name: layers = "amnion"
        elif "chorion" in subsample_name: layers = "chorion"
        
        # labored/unlabored
        pregnancy = "C_section" if "C_section" in subsample_name else "labored"
        
        # POPULATE DATAFRAME
        data = {"sample": [sample], "subsample": [subsample_name], "location" : [location] , "layers": [layers], "pregnancy":pregnancy, "max_pressure": [max_pressure]}
        df = df.append(pd.DataFrame(data=data))
    
    # amniochorion and pericervical
    # list_path_amniochorion_pericervical = list(filter(re.compile("pericervical").search, list_path_amniochorion))[0]

#%%
import pandas as pd 
import numpy as np
import holoviews as hv 
from holoviews import opts
# hv.extension("bokeh")
hv.extension("matplotlib")

import matplotlib as mpl
mpl.rcParams["figure.dpi"] =300
 
#%% Amniochorion --> periplacental vs pericervical

#amniochorion 
df = df.dropna() # if you don't do this it won't plot data
boxwhisker = hv.BoxWhisker(df, ["location", "layers"], "max_pressure", label="Max Pressure kPa" )
boxwhisker.opts(xrotation=90)


for layers in ["amnion", "amniochorion", "chorion"]:
    pass
    for loc in ["pericervical", "periplacental"]:
        pass

        # 
        df_loc = df[df["location"]== loc]
        df_loc_layer = df_loc[df_loc["layers"] == layers]
        print(f"{loc} | {layers}  : {len(df_loc_layer)}")

#%%
hv.render(boxwhisker, backend="matplotlib") # plot data

#%%


table = hv.Table(boxwhisker)
hv.render(table, backend="matplotlib") # plot data

#%%

groups = [chr(65+g) for g in np.random.randint(0, 3, 200)]
boxwhisker = hv.BoxWhisker((groups, np.random.randint(0, 5, 200), np.random.randn(200)),
              ['Group', 'Category'], 'Value').sort()
boxwhisker.opts(
    opts.BoxWhisker(box_color='white', height=400, show_legend=False, whisker_color='gray', width=600))
    

