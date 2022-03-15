from pathlib import Path
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl 
mpl.rcParams["figure.dpi"] = 300
import numpy as np
from sklearn.model_selection import ParameterGrid
from pprint import pprint
# plot
import holoviews as hv
hv.extension("bokeh")
from holoviews import opts
from tqdm import tqdm
import re
from processing_pad_relaynet_for_merging import pad_relaynet_for_merging

path_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files")



# find exports to merge into df
list_pressure_csv_files = list(path_dataset.rglob("*_Pressure_Apex.csv"))
list_thickness_csv_files = list(path_dataset.rglob("*_thickness.csv"))

# create tupples only of samples that have pressure and thickness exports
list_tuples_of_matching_files = []
for path_pressure_file in list_pressure_csv_files:
    pass
    base_name = path_pressure_file.stem.rsplit("_",2)[0]
    for path_thickness_file in list_thickness_csv_files:
        pass
        if f"{base_name}_thickness.csv" in str(path_thickness_file):
            list_tuples_of_matching_files.append({
                "path_pressure_csv" : path_pressure_file,
                    "path_thickness_csv" : path_thickness_file
                })

# double check files match
for d in list_tuples_of_matching_files:
    pass
    assert d['path_pressure_csv'].stem.split("_",2)[0] == \
        d['path_thickness_csv'].stem.split("_", 1)[0], "pressure and thickness files don't match"

#%%

dict_params = { # iterate through (7.5-15) up to (max, 17.8)
    "loaded_lower_bound" : np.arange(7.5, 15.5, step=0.5),
    "loaded_upper_bound" : ["max", 17.8]
    }

list_combinations = list(ParameterGrid(dict_params))

# holoviews_apex_pressure = None
# overview_holoviews_apex_pressure = None
    

# iterate through every sample found
for pos, dict_paths in tqdm(enumerate(list_tuples_of_matching_files[:])): # for each parameter, iterate through the feature csv files
    pass
    file_path = dict_paths["path_pressure_csv"]
    print(file_path.stem)

    ## grid search
    dict_dataset = {}
    for dict_params in list_combinations: #iterate through parameters
        pass
    
    
        ## AGGREGATE PRESSURE AND THICKNESS FILES ON THE FLY
        # _Pressure_Apex.csv and _thickness.csv
        df_pressure = pd.read_csv(dict_paths["path_pressure_csv"], names=["Apex Rise","Pressure"])
        df_thickness = pd.read_csv(dict_paths["path_thickness_csv"])
        
        ## if not same lengths
        if not len(df_pressure) == len(df_thickness):
           df_thickness = pad_relaynet_for_merging(df_pressure, df_thickness)
        
        df = pd.concat([df_pressure, df_thickness], axis=1)
        df.index = np.arange(1, len(df_thickness)+1)
        df.index.names = ["frame_number"] 

        df = df.dropna() # drop NA values
        
        
        
        ### add column of combined thicknesses 
        df["amnion_spongy_chorion-thickness"] = df["amnion-thickness"] + df["spongy-thickness"] + df["chorion-thickness"]
            
        # get indices from pressure array 
        def get_region_indices(data, val_lower, val_upper):
            # extract regions based on first instance of lower and upper bounds
            idx_lower = np.argwhere(data > val_lower).squeeze()
            if idx_lower.size == 0:
                print("no value greater than lower bound")
                return None
            
            #get firstvalue
            idx_lower = idx_lower[0] if idx_lower.ndim > 0 else int(idx_lower)
            
            idx_upper = np.argwhere(data > val_upper).squeeze()
            if idx_upper.size == 0: # no values larger than upper, find idx of max value 
                idx_upper = np.argwhere(data == data.max()).squeeze()
            
            #get first value
            idx_upper = idx_upper[0] if idx_upper.ndim > 0 else int(idx_upper) # get first value
                
            return idx_lower, idx_upper
        
        
        ## GET TOE/LOADED INDICES FROM PRESSURE CURVE
        # toe range
        thresh_toe_low = 0.5
        thresh_toe_high = 5
        idx_toe = get_region_indices(df["Pressure"].values,
                                     thresh_toe_low,
                                     thresh_toe_high)
        
        # loaded region range 
        thresh_loaded_low = dict_params["loaded_lower_bound"]
        if dict_params["loaded_upper_bound"] == "max":
            thresh_loaded_high = np.max(df["Pressure"].values) # or 17.8
        else:
            thresh_loaded_high = dict_params["loaded_upper_bound"]

            
        if thresh_loaded_high < thresh_loaded_low:
            print("Error Loaded region: range error. lower boundary greater than upper boundary")
            print(f"{file_path}")
            continue

        idx_loaded = get_region_indices(df["Pressure"].values,
                                     thresh_loaded_low,
                                     thresh_loaded_high)
        if idx_loaded == None:
            print(f"skipping | {file_path.stem}")
            continue

        
        # ## visualize regions 
        # # plot regions
        # plt.title(f"Apex Rise vs Pressure \n {pos} | {file_path.stem}")
        # plt.plot(df["Apex Rise"],df["Pressure"])
        # #toe region
        # plt.plot(df["Apex Rise"][idx_toe[0]:idx_toe[1]], 
        #           df["Pressure"][idx_toe[0]:idx_toe[1]], label="toe region",
        #           )
        # # loaded region
        # plt.plot(df["Apex Rise"][idx_loaded[0]:idx_loaded[1]],
        #           df["Pressure"][idx_loaded[0]:idx_loaded[1]] , label="loaded region",
        #           )
        # plt.legend()
        # plt.xlabel("Apex Rise [mm]")
        # plt.ylabel("Pressure [kPa]")
        # plt.show()
        
        # ## HOLOVIEWS OVERLAY OBJECT
        # kdims = ["Apex Rise"]
        # vdims = ["Pressure"]
        # hv_apex_rise_pressure = hv.Scatter((df["Apex Rise"], df["Pressure"]),
        #                                    kdims=kdims, vdims=vdims,
        #                                    label="Apex Rise vs Pressure")
        # hv_apex_rise_pressure.opts(color="g")
        # holoviews_toe = hv.Scatter((df["Apex Rise"][idx_toe[0]:idx_toe[1]], df["Pressure"][idx_toe[0]:idx_toe[1]]), 
        #                            kdims=kdims, vdims=vdims, 
        #                            label=f"Toe | Range {thresh_toe_low} to {thresh_toe_high}")
        # holoviews_toe.opts(color="b")
        # holoviews_loaded = hv.Scatter((df["Apex Rise"][idx_loaded[0]:idx_loaded[1]], df["Pressure"][idx_loaded[0]:idx_loaded[1]]),
        #                              kdims=kdims, vdims=vdims, 
        #                               label= f"loaded | Range {thresh_loaded_low} to {thresh_loaded_high} ")
        # holoviews_loaded.opts(color="r")
        
        # holoviews_apex_pressure = hv_apex_rise_pressure * holoviews_toe * holoviews_loaded
        
        # if overview_holoviews_apex_pressure is None:
        #     overview_holoviews_apex_pressure = holoviews_apex_pressure
        # else:
        #     # overview_holoviews_apex_pressure *= holoviews_apex_pressure
        #     overview_holoviews_apex_pressure *= holoviews_apex_pressure


        # initialize dict for storage
        sample_name = f"toe_{thresh_toe_low}-{thresh_toe_high}_loaded_{thresh_loaded_low}-{thresh_loaded_high}"
        dict_dataset[sample_name] = {} # add entry for this sample
        # dict_dataset[sample_name]["sample_path"] = file_path
        
        dict_dataset[sample_name]["threshold_toe_low"] = thresh_toe_low
        dict_dataset[sample_name]["threshold_toe_high"] = thresh_toe_high
        
        dict_dataset[sample_name]["threshold_loaded_low"] = thresh_loaded_low
        dict_dataset[sample_name]["threshold_loaded_high"] = thresh_loaded_high #dict_params["loaded_upper_bound"]
    
        ### COMPUTE AVEARAGE THICKNESS AT RANGES
        ## toe thickness    
        dict_dataset[sample_name]["avg_amnion_toe_thickness"] = np.mean(df["amnion-thickness"][idx_toe[0]:idx_toe[1]])
        dict_dataset[sample_name]["avg_spongy_toe_thickness"] = np.mean(df["spongy-thickness"][idx_toe[0]:idx_toe[1]])
        dict_dataset[sample_name]["avg_chorion_toe_thickness"] = np.mean(df["chorion-thickness"][idx_toe[0]:idx_toe[1]])
        dict_dataset[sample_name]["avg_decidua_toe_thickness"] = np.mean(df["decidua-thickness"][idx_toe[0]:idx_toe[1]])
        dict_dataset[sample_name]["avg_amnion_spongy_chorion_toe_thickness"] = np.mean(df["amnion_spongy_chorion-thickness"][idx_toe[0]:idx_toe[1]])
    
    
        ## loaded thicknesses
        dict_dataset[sample_name]["avg_amnion_loaded_thickness"] = np.mean(df["amnion-thickness"][idx_loaded[0]:idx_loaded[1]])
        dict_dataset[sample_name]["avg_spongy_loaded_thickness"] = np.mean(df["spongy-thickness"][idx_loaded[0]:idx_loaded[1]])
        dict_dataset[sample_name]["avg_chorion_loaded_thickness"] = np.mean(df["chorion-thickness"][idx_loaded[0]:idx_loaded[1]])
        dict_dataset[sample_name]["avg_decidua_loaded_thickness"] = np.mean(df["decidua-thickness"][idx_loaded[0]:idx_loaded[1]])
        dict_dataset[sample_name]["avg_amnion_spongy_chorion_loaded_thickness"] = np.mean(df["amnion_spongy_chorion-thickness"][idx_loaded[0]:idx_loaded[1]])
    
        
        ### DETERMINE OTHER PARAMETERS 
        # term
        csv_filename = file_path.stem 
        dict_dataset[sample_name]["birth_type"] = "unlabored" if "C_section" in csv_filename \
            else "labored"
            
        # Location
        if "cervical" in csv_filename:
            dict_dataset[sample_name]["location"] = "pericervical"  
        elif "placental" in csv_filename:
            dict_dataset[sample_name]["location"] = "periplacental" 
        else: dict_dataset[sample_name]["location"] = np.NaN
        
        # layers (Amniochorion, amnion and chorion)
        if "amniochorion" in csv_filename:
            dict_dataset[sample_name]["layers"] = "amniochorion"
        elif "amnion" in csv_filename:
            dict_dataset[sample_name]["layers"] = "amnion"
        elif "chorion" in csv_filename:
            dict_dataset[sample_name]["layers"] = "chorion"
        else: dict_dataset[sample_name]["layers"] = np.NaN

        # SAVE DATAFRAME FOR SAMPLE
        #%%
    path_sample_output = file_path.parent
    filename = file_path.stem.rsplit("_", 2)[0]
    df_thick = pd.DataFrame(dict_dataset).transpose()
    df_thick.index.name = "sample_name"
    df_thick.to_csv(path_sample_output / f"{filename}_toe_loaded_thicknesses.csv")
        
