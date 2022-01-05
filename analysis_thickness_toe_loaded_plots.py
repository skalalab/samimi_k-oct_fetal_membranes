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

path_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files")

list_feature_csv_files = list(path_dataset.rglob("*features.csv"))
#%%


# iterate through (7.5-15) up to (max, 17.8)

dict_params = {
    "loaded_lower_bound" : np.arange(7.5, 15.5, step=0.5),
    "loaded_upper_bound" : ["max", 17.8]
    }

list_combinations = list(ParameterGrid(dict_params))



for dict_params in list_combinations: #iterate through parameters
    pass
    
    holoviews_apex_pressure = None
    overview_holoviews_apex_pressure = None
    dict_dataset = {}
    for pos, file_path in enumerate(list_feature_csv_files): # for each parameter, iterate through the feature csv files
        pass
        df = pd.read_csv(file_path)
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
        
        ## holoviews overlay object
        hv_apex_rise_pressure = hv.Scatter((df["Apex Rise"], df["Pressure"]),
                                           kdims=["Pressure"], 
                                           vdims=["Apex Rise"], 
                                           label="Apex Rise vs Pressure")
        holoviews_toe = hv.Scatter((df["Apex Rise"][idx_toe[0]:idx_toe[1]], df["Pressure"][idx_toe[0]:idx_toe[1]]), 
                                   kdims=["Pressure"], 
                                   vdims=["Apex Rise"],
                                   label=f"Toe | Range {thresh_toe_low} to {thresh_toe_high}")
        holoviews_loaded = hv.Scatter((df["Apex Rise"][idx_loaded[0]:idx_loaded[1]], df["Pressure"][idx_loaded[0]:idx_loaded[1]]),
                                      kdims=["Pressure"],
                                      vdims=["Apex Rise"],
                                      label= f"loaded | Range {thresh_loaded_low} to {thresh_loaded_high} ")
        
        holoviews_apex_pressure = hv_apex_rise_pressure * holoviews_toe * holoviews_loaded
        
        if overview_holoviews_apex_pressure is None:
            overview_holoviews_apex_pressure = holoviews_apex_pressure
        else:
            overview_holoviews_apex_pressure *= holoviews_apex_pressure
        
    
        # initialize dict
        sample_name = file_path.stem
        dict_dataset[sample_name] = {} # add entry for this sample
        dict_dataset[sample_name]["sample_path"] = file_path
        
        dict_dataset[sample_name]["threshold_toe_low"] = thresh_toe_low
        dict_dataset[sample_name]["threshold_toe_high"] = thresh_toe_high
        
        dict_dataset[sample_name]["threshold_loaded_low"] = thresh_loaded_low
        dict_dataset[sample_name]["threshold_loaded_high"] = dict_params["loaded_upper_bound"]
    
        ### COMPUTE AVEARAGE THICKNESS AT RANGES
        ## toe thickness    
        dict_dataset[sample_name]["avg_amnion_toe_thickness"] = np.mean(df["amnion-thickness"][idx_toe[0]:idx_toe[1]])
        dict_dataset[sample_name]["avg_spongy_toe_thickness"] = np.mean(df["spongy-thickness"][idx_toe[0]:idx_toe[1]])
        dict_dataset[sample_name]["avg_chorion_toe_thickness"] = np.mean(df["chorion-thickness"][idx_toe[0]:idx_toe[1]])
        dict_dataset[sample_name]["avg_amnion_spongy_chorion_toe_thickness"] = np.mean(df["amnion_spongy_chorion-thickness"][idx_toe[0]:idx_toe[1]])
    
    
        ## loaded thicknesses
        dict_dataset[sample_name]["avg_amnion_loaded_thickness"] = np.mean(df["amnion-thickness"][idx_loaded[0]:idx_loaded[1]])
        dict_dataset[sample_name]["avg_spongy_loaded_thickness"] = np.mean(df["spongy-thickness"][idx_loaded[0]:idx_loaded[1]])
        dict_dataset[sample_name]["avg_chorion_loaded_thickness"] = np.mean(df["chorion-thickness"][idx_loaded[0]:idx_loaded[1]])
        dict_dataset[sample_name]["avg_amnion_spongy_chorion_loaded_thickness"] = np.mean(df["amnion_spongy_chorion-thickness"][idx_loaded[0]:idx_loaded[1]])
    
        
        ### DETERMINE OTHER PARAMETERS 
        # term    
        dict_dataset[sample_name]["term"] = "unlabored" if "C_section" in sample_name \
            else "labored"
        # Location
        if "pericervical" in sample_name:
            dict_dataset[sample_name]["location"] = "pericervical"  
        elif "periplacental" in sample_name:
            dict_dataset[sample_name]["location"] = "periplacental" 
        
        # layers (Amniochorion, amnion and chorion)
        if "amniochorion" in sample_name:
            dict_dataset[sample_name]["layers"] = "amniochorion"
        elif "amnion" in sample_name:
            dict_dataset[sample_name]["layers"] = "amnion"
        elif "chorion" in sample_name:
            dict_dataset[sample_name]["layers"] = "chorion"
        else: dict_dataset[sample_name]["layers"] = np.NaN
        
#%%
    # convert to dataframe
    df_thick = pd.DataFrame(dict_dataset).transpose()
    
    df_thick = df_thick.dropna()

    
    path_output = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\figures\thickness_toe_loaded")
    
    # toe
    kdims = [("term", "Term"),("location","Location")]
   
    bw_toe_amnion = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_amnion_toe_thickness","Avg Thickness (px)")], label="Amnion Toe")
    bw_toe_spongy = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_spongy_toe_thickness","Avg Thickness (px)")], label="Spongy Toe")
    bw_toe_chorion = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_chorion_toe_thickness","Avg Thickness (px)")], label="Chorion Toe")
    bw_toe_combined = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_amnion_spongy_chorion_toe_thickness","Avg Thickness (px)")], label="Amion,Spongy,Chorion Toe")


    layout_toe = holoviews_apex_pressure + bw_toe_amnion + bw_toe_spongy + bw_toe_chorion + bw_toe_combined

    # LOADED 
    bw_loaded_amnion = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_amnion_loaded_thickness","Avg Thickness (px)")], label="Amnion Loaded")
    bw_loaded_spongy = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_spongy_loaded_thickness","Avg Thickness (px)")], label="Spongy Loaded")
    bw_loaded_chorion = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_chorion_loaded_thickness","Avg Thickness (px)")], label="Chorion Loaded")
    bw_loaded_combined = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_amnion_spongy_chorion_loaded_thickness","Avg Thickness (px)")], label="Amnion,Spongy,Chorion Loaded")

    layout_loaded =  overview_holoviews_apex_pressure + bw_loaded_amnion + bw_loaded_spongy + bw_loaded_chorion + bw_loaded_combined
    
    #global options
    overlay = layout_toe + layout_loaded
    overlay.opts(        
        opts.BoxWhisker(width=500, height=500, tools=["hover"], legend_position='right'),
        opts.Scatter(width=500, height=500, tools=["hover"], legend_position='top_left', alpha=1),
    ).cols(5)
    

    str_loaded_range = f"{df_thick['threshold_loaded_low'][0]}_to_{df_thick['threshold_loaded_high'][0]}"
    hv.save(overlay, path_output / f"thickness_loaded_range_{str_loaded_range}.html")
    df_thick.to_csv(path_output / f"thickness_loaded_range_{str_loaded_range}.csv")
    
        
