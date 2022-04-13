from pathlib import Path
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from numpy.linalg import lstsq
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
from sklearn.model_selection import ParameterGrid
import holoviews as hv
from holoviews import opts
hv.extension("bokeh")
from tqdm import tqdm

path_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files")
                
# list_path_features_csv = list(path_dataset.rglob("*amniochorion*features.csv"))
list_path_features_csv = list(path_dataset.rglob("*_Pressure_Apex.csv"))
#%%

dict_params = {
    "loaded_lower_bound" : np.arange(7.5, 15.5, step=0.5),
    "loaded_upper_bound" : ["max", 17.8]
    }

list_combinations = list(ParameterGrid(dict_params))

# store values 
# holoviews_apex_pressure = None
# overview_holoviews_apex_pressure = None
list_toe_greater_than_loaded = []


    
for pos, path_csv in tqdm(enumerate(list_path_features_csv[:-1])): # iterate through
    pass
    print(path_csv.stem)
    ## grid search
    dict_tension_strain_modulus = {}
    dict_frame_by_frame_tension_strain = {}
    for dict_params in list_combinations: #iterate through parameters
        pass
    
        df = pd.read_csv(path_csv, names=["Apex Rise", "Pressure"])
        df = df.reset_index()
        df = df.rename(columns={"index":"frame_number"})
        df["frame_number"] = df["frame_number"] + 1 # first frame is 1 not zero
        df = df.dropna()
        df = df.reset_index(drop=True) # reset index
        df_pressure_apex = df
        
        #%% Extract indices for toe and loading region
        
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
            print(f"{path_csv}")
            list_toe_greater_than_loaded.append(path_csv)
            # plot invalid region
            # plt.title(f"INVALID RANGE \n Apex Rise vs Pressure \n {pos} | {path_csv.stem}")
            # plt.plot(df_pressure_apex["Apex Rise"],df_pressure_apex["Pressure"])
            # plt.xlabel("Apex Rise [mm]")
            # plt.ylabel("Pressure [kPa]")
            # plt.show()
            continue
        
        idx_loaded = get_region_indices(df_pressure_apex["Pressure"].values,
                                     thresh_loaded_low,
                                     thresh_loaded_high)
        if idx_loaded == None or idx_loaded[0]==idx_loaded[1]: # range is the same value
            print(f"skipping | {path_csv.stem}")
            continue
        
        ## HOLOVIEWS OVERLAY OBJECT
        # kdims = ["Apex Rise"]
        # vdims = ["Pressure"]
        # hv_apex_rise_pressure = hv.Scatter((df["Apex Rise"], df["Pressure"]),
        #                                    kdims=kdims, vdims=vdims, 
        #                                    label="Apex Rise vs Pressure")
        # holoviews_toe = hv.Scatter((df["Apex Rise"][idx_toe[0]:idx_toe[1]], df["Pressure"][idx_toe[0]:idx_toe[1]]), 
        #                            kdims=kdims, vdims=vdims, 
        #                            label=f"Toe | Range {thresh_toe_low} to {thresh_toe_high}")
        # holoviews_loaded = hv.Scatter((df["Apex Rise"][idx_loaded[0]:idx_loaded[1]], df["Pressure"][idx_loaded[0]:idx_loaded[1]]),
        #                               kdims=kdims, vdims=vdims, 
        #                               label= f"loaded | Range {thresh_loaded_low} to {thresh_loaded_high} ")
        
        # holoviews_apex_pressure = hv_apex_rise_pressure * holoviews_toe * holoviews_loaded
        
        # if overview_holoviews_apex_pressure is None:
        #     overview_holoviews_apex_pressure = holoviews_apex_pressure
        # else:
        #     overview_holoviews_apex_pressure *= holoviews_apex_pressure
        
        # # plot regions
        # plt.title(f"Apex Rise vs Pressure \n {pos} | {path_csv.stem}")
        # plt.plot(df_pressure_apex["Apex Rise"],df_pressure_apex["Pressure"])
        # #toe region
        # plt.plot(df_pressure_apex["Apex Rise"][idx_toe[0]:idx_toe[1]], 
        #           df_pressure_apex["Pressure"][idx_toe[0]:idx_toe[1]], label="toe region",
        #           )
        # # loaded region
        # plt.plot(df_pressure_apex["Apex Rise"][idx_loaded[0]:idx_loaded[1]],
        #           df_pressure_apex["Pressure"][idx_loaded[0]:idx_loaded[1]] , label="loaded region",
        #           )
        # plt.legend()
        # plt.xlabel("Apex Rise [mm]")
        # plt.ylabel("Pressure [kPa]")
        # plt.show()
        
        #%% Tension-Strain Analysis
        
        initial_apex = 6   # in [mm]
        initial_pressure = 0 # kPa
        device_radius = 15 # radius of  device that holds the membrane
        meters_to_mm = 1000
        
        initial_radius = (initial_apex**2 + device_radius**2)/(2 * initial_apex)   # in [mm]
        
        # add offset initial offset to apex and set starting value to initial offset
        apex = initial_apex + np.asarray(df_pressure_apex["Apex Rise"]) # add offset
        apex = np.insert(apex, 0, initial_apex) # add initial position of zero
        
        # set initial pressure to zero
        pressure = np.asarray(df_pressure_apex["Pressure"])
        pressure = np.insert(pressure,0, initial_pressure)
        
        # m ==>mm /2
        radius = (apex**2 + device_radius**2) / (2 * apex)   # in [mm]  *****
        tension = (pressure * radius) / (2 * meters_to_mm);   # in [N/mm]
        strain = apex / radius # *****
        
        #%%
        # y = mx + c==> y = Ap
        # A = [[x 1]] and p = [[m], [c]] 
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
        # strain, concatenate a column of ones for matrix opearation
        
        def best_fit_line(A, y):
            A = np.concatenate( 
                    (A[:,np.newaxis],
                    np.ones((len(A),1))), 
                axis=1)
            m,c = lstsq(A, y, rcond=None)[0]
            return m, c  # m=slope, c=yintercept
        
        #%% TENSIONS-STRAIN PLOTS 
        
        tension_meters = tension*1000
        
        # TOE REGION
        toe_strain = strain[idx_toe[0]:idx_toe[1]]
        toe_tension = tension_meters[idx_toe[0]:idx_toe[1]]    
        toe_slope, toe_y_int = best_fit_line(toe_strain, toe_tension)
        x_toe = toe_strain
       
        
        # LOADED REGION
        loaded_strain = strain[idx_loaded[0]:idx_loaded[1]]
        loaded_tension = tension_meters[idx_loaded[0]:idx_loaded[1]]
        loaded_slope, loaded_y_int = best_fit_line(loaded_strain, loaded_tension)
        x_loaded = loaded_strain
        
        ## PLOT ODD SAMPLES TOE SLOPE > LOADED SLOPE
        # if toe_slope > loaded_slope:
            
        #     plt.title(f"Tension vs Strain \n {path_csv.stem}")
            
        #     plt.plot(strain, tension_meters)
            
        #     # toe
        #     plt.plot(toe_strain, toe_tension , color="r")
        #     plt.plot(x_toe, toe_slope*x_toe + toe_y_int, label="toe", color="r")
        #     plt.text(np.min(toe_strain), np.mean(toe_tension), f"Tension Modulus\n={int(toe_slope)} [N/m] ",color="r")
        #     # loaded
        #     plt.plot(loaded_strain, loaded_tension , color="g")
        #     plt.plot(x_loaded, loaded_slope*x_loaded + loaded_y_int, label="loaded", color="g")
        #     plt.text(np.min(loaded_strain)*.9, np.mean(loaded_tension), f"Tension Modulus\n={int(loaded_slope)} [N/m] ",color="g")
            
        #     plt.legend()
        #     plt.xlabel("Strain")
        #     plt.ylabel("Tension [N/m]")
        #     plt.show()

        ## SAVE VALUES INTO A DICTIONARY
        # 
        sample_name = f"toe_{thresh_toe_low}-{thresh_toe_high}_loaded_{thresh_loaded_low}-{thresh_loaded_high}"

        dict_tension_strain_modulus[sample_name] = {} # add entry for this sample
        ## save frame by frame tension/strain values

        #save individual csv files?
        # df_frame_by_frame_tension_strain = pd.DataFrame(
        #     {"tension" : tension[1:],
        #      "strain" : strain[1:]})
        # df.to_csv(path_features_all / f"{path_csv.stem}_all.csv")

        
        dict_tension_strain_modulus[sample_name]["tension"] = tension[1:] # we inserted 1st value
        dict_tension_strain_modulus[sample_name]["strain"] =  strain[1:]# we inserted 1st value
        dict_tension_strain_modulus[sample_name]["frame_indices"] =  df_pressure_apex["frame_number"].values
        
        # save max values
        dict_tension_strain_modulus[sample_name]["max_tension"] = np.max(tension[1:])
        dict_tension_strain_modulus[sample_name]["max_strain"] = np.max(strain[1:])
        dict_tension_strain_modulus[sample_name]["max_apex"] = np.max(apex)
        
        # frame ranges 
        # dict_tension_strain_modulus[sample_name]["toe_frame_range"] = idx_toe
        # dict_tension_strain_modulus[sample_name]["loaded_frame_range"] = idx_loaded
        
        # df = df.set_index("frame_number", drop=True)
        #df.to_csv(path_features_all / f"{path_csv.stem}_all.csv")
        
        ##         
        
        # dict_tension_strain_modulus[sample_name]["sample_path"] = path_csv
        
        ## save thresholds
        dict_tension_strain_modulus[sample_name]["threshold_toe_low"] = thresh_toe_low
        dict_tension_strain_modulus[sample_name]["threshold_toe_high"] = thresh_toe_high
        
        dict_tension_strain_modulus[sample_name]["threshold_loaded_low"] = thresh_loaded_low
        dict_tension_strain_modulus[sample_name]["threshold_loaded_high"] = dict_params["loaded_upper_bound"]
        
        
        csv_filename = path_csv.stem 
        # TERM    
        dict_tension_strain_modulus[sample_name]["birth_type"] = "unlabored" if "C_section" in csv_filename \
            else "labored"
        
        # save tension modulus
        dict_tension_strain_modulus[sample_name]["toe_modulus"] = toe_slope
        dict_tension_strain_modulus[sample_name]["loaded_modulus"] = loaded_slope
        
        # Location
        if "pericervical" in csv_filename:
            dict_tension_strain_modulus[sample_name]["location"] = "pericervical"  
        elif "periplacental" in csv_filename:
            dict_tension_strain_modulus[sample_name]["location"] = "periplacental" 
        else:
            dict_tension_strain_modulus[sample_name]["location"] = ""
        
        # LAYERS (Amniochorion, amnion and chorion)
        if "amniochorion" in csv_filename:
            dict_tension_strain_modulus[sample_name]["layers"] = "amniochorion"
        elif "amnion" in csv_filename:
            dict_tension_strain_modulus[sample_name]["layers"] = "amnion"
        elif "chorion" in csv_filename:
            dict_tension_strain_modulus[sample_name]["layers"] = "chorion"
        else: dict_tension_strain_modulus[sample_name]["layers"] = np.NaN
    
    #%%
    path_output = path_csv.parent
    filename = path_csv.stem.rsplit("_", 2)[0]
    
    df_tension_strain = pd.DataFrame(dict_tension_strain_modulus).transpose()
    df_tension_strain.index.name = "sample_name"
    
    if len(dict_tension_strain_modulus) != 0: #only save non empty dictionaries
        df_tension_strain.to_csv(path_output / f"{filename}_toe_loaded_tension_strain.csv")
    
    ## separate odd samples
    # df_loaded_tension_greater_than_toe = df_tension_strain[df_tension_strain["loaded_tension"] > df_tension_strain["toe_tension"]]
    # df_funky = df_tension_strain[df_tension_strain["loaded_tension"] < df_tension_strain["toe_tension"]]
    # if len(df_funky) > 0:
    #     pass
        #df_funky.to_csv(path_output / f"tension_strain_range_{str_loaded_range}_toe_greater_loaded.csv")

    


