from pathlib import Path
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl 
mpl.rcParams["figure.dpi"] = 300
import numpy as np
from sklearn.model_selection import ParameterGrid
from pprint import pprint

path_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files")

list_feature_csv_files = list(path_dataset.rglob("*features.csv"))
#%%


# iterate through (7.5-15) up to (max, 17.8)

dict_params = {
    "loaded_lower_bound" : np.arange(7.5, 15.5, step=0.5),
    "loaded_upper_bound" : ["max", 17.8]
    }

list_combinations = list(ParameterGrid(dict_params))

for dict_params in list_combinations: #iterate through the ranges 
    pass

    dict_dataset = {}
    for pos, file_path in enumerate(list_feature_csv_files): # iterate through the feature csv files
        pass
        df = pd.read_csv(file_path)
        df = df.dropna() # drop NA values
        
        ### add column of combined thicknesses 
        df["amnion_spongy_chorion-area"] = df["amnion-area"] + df["spongy-area"] + df["chorion-area"]
            
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
            
            #get firstvalue
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

        
        ## visualize regions 
        # plot regions
        plt.title(f"Apex Rise vs Pressure \n {pos} | {file_path.stem}")
        plt.plot(df["Apex Rise"],df["Pressure"])
        #toe region
        plt.plot(df["Apex Rise"][idx_toe[0]:idx_toe[1]], 
                  df["Pressure"][idx_toe[0]:idx_toe[1]], label="toe region",
                  )
        # loaded region
        plt.plot(df["Apex Rise"][idx_loaded[0]:idx_loaded[1]],
                  df["Pressure"][idx_loaded[0]:idx_loaded[1]] , label="loaded region",
                  )
        plt.legend()
        plt.xlabel("Apex Rise [mm]")
        plt.ylabel("Pressure [kPa]")
        plt.show()
        
    
        # initialize dict
        sample_name = file_path.stem
        dict_dataset[sample_name] = {} # add entry for this sample
        dict_dataset[sample_name]["sample_path"] = file_path
        
        dict_dataset[sample_name]["threshold_toe_low"] = thresh_toe_low
        dict_dataset[sample_name]["threshold_toe_high"] = thresh_toe_high
        
        dict_dataset[sample_name]["threshold_loaded_low"] = thresh_loaded_low
        dict_dataset[sample_name]["threshold_loaded_high"] = thresh_loaded_high
    
        ### GET AVERAGE THICKNESSES
        ## toe thickness    
        dict_dataset[sample_name]["avg_amnion_toe_thickness"] = np.mean(df["amnion-area"][idx_toe[0]:idx_toe[1]])
        dict_dataset[sample_name]["avg_spongy_toe_thickness"] = np.mean(df["spongy-area"][idx_toe[0]:idx_toe[1]])
        dict_dataset[sample_name]["avg_chorion_toe_thickness"] = np.mean(df["chorion-area"][idx_toe[0]:idx_toe[1]])
        dict_dataset[sample_name]["avg_amnion_spongy_chorion_toe_thickness"] = np.mean(df["amnion_spongy_chorion-area"][idx_toe[0]:idx_toe[1]])
    
    
        ## loaded thicknesses
        dict_dataset[sample_name]["avg_amnion_loaded_thickness"] = np.mean(df["amnion-area"][idx_loaded[0]:idx_loaded[1]])
        dict_dataset[sample_name]["avg_spongy_loaded_thickness"] = np.mean(df["spongy-area"][idx_loaded[0]:idx_loaded[1]])
        dict_dataset[sample_name]["avg_chorion_loaded_thickness"] = np.mean(df["chorion-area"][idx_loaded[0]:idx_loaded[1]])
        dict_dataset[sample_name]["avg_amnion_spongy_chorion_loaded_thickness"] = np.mean(df["amnion_spongy_chorion-area"][idx_loaded[0]:idx_loaded[1]])
    
        
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
    df = pd.DataFrame(dict_dataset).transpose()
    
    df = df.dropna()
    # plot
    import holoviews as hv
    hv.extension("bokeh", "matplotlib")
    
    path_output = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\figures\thickness_toe_loaded")
    
    # toe
    kdims = [("term", "Term"),("location","Location")]
    vdims = [("avg_amnion_toe_thickness","Thickness (px)")]
    
    violin_toe_amnion = hv.BoxWhisker(df, kdims , vdims, label="Average Amnion Toe Thicknesses")
    
    
    str_loaded_range = f"{df['threshold_loaded_low'][0]}_to_{df['threshold_loaded_high'][0]}"
    hv.save(violin_toe_amnion, path_output / f"test_{str_loaded_range}.html")
    
    
    #global options
    opts(
        opts.BoxWhisker(width=800, height=800)
    )
    
    
    
    # loaded 
    
    'avg_amnion_loaded_thickness',
    'avg_amnion_toe_thickness',
    'avg_chorion_loaded_thickness',
    'avg_chorion_toe_thickness',
    'avg_spongy_loaded_thickness',
    'avg_spongy_toe_thickness',
    
    'avg_amnion_spongy_chorion_loaded_thickness',
    'avg_amnion_spongy_chorion_toe_thickness',
    
    #4x2 
    ### toe loaded 
    # Thicknesses
        # chorion
        # spongy
        # amnion 
        # combined
    
    
    amnion toe thickness
    #location
    #term
    
    # save 
        
        
