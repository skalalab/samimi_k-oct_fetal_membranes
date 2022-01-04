from pathlib import Path
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl 
# mpl.use("TKAgg")
mpl.rcParams["figure.dpi"] = 300
import numpy as np


path_dataset_soft_tissue_lab = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\dataset_for_soft_tissue_lab")

list_feature_csv_files = list(path_dataset_soft_tissue_lab.rglob("*features.csv"))
#%%


dict_dataset = {}
for pos, file_path in enumerate(list_feature_csv_files):
    pass
    df = pd.read_csv(file_path)
    df = df.dropna() # drop NA values
    
    ### add column of combined thicknesses 
    df["amnion_spongy_chorion-area"] = df["amnion-area"] + df["spongy-area"] + df["chorion-area"]
        
    # get indices from pressure array 
    def get_region_indices(data, val_lower, val_upper):
        # extract regions based on first instance of lower and upper bounds
        idx_lower = np.argwhere(data > val_lower).squeeze()[0]
        idx_upper = np.argwhere(data > val_upper).squeeze()
        if idx_upper.size == 0: # no values larger than upper, find idx of max value 
            idx_upper = np.argwhere(data == data.max()).squeeze()
            if idx_upper.shape == (): # single value 0D array returned
                idx_upper = int(idx_upper)
            else: # if multiple max values found
                idx_upper = idx_upper[0]
        elif idx_upper.size == 1: #single upper idx value found
            idx_upper = int(idx_upper)
        else:
            idx_upper = idx_upper[0]
        return idx_lower, idx_upper
    
    
    ## GET TOE/LOADED INDICES FROM PRESSURE CURVE
    # toe range
    thresh_toe_low = 0.5
    thresh_toe_high = 5
    idx_toe = get_region_indices(df["Pressure"].values,
                                 thresh_toe_low,
                                 thresh_toe_high)
    # loaded region range 
    thresh_loaded_low = 7.5
    thresh_loaded_high = np.max(df["Pressure"].values) # or 17.8
    
        
    if thresh_loaded_high < thresh_loaded_low:
        print(f"Error Loaded region: range error. lower boundary greater than upper boundary")
        print(f"{file_path}")
        continue
    
    idx_loaded = get_region_indices(df["Pressure"].values,
                                 thresh_loaded_low,
                                 thresh_loaded_high)

    
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

    ### GET AVERAGE THICKNESSES
    ## toe thicknesse    ### GET AVERAGE THICKNESSESs
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
    
# convert to dataframe
df = pd.DataFrame(dict_dataset).transpose()
# plot 
# save 
    
        
