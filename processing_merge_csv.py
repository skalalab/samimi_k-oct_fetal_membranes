#%% merge matlab outputs and python into one csv file

from pathlib import Path
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from processing_pad_relaynet_for_merging import pad_relaynet_for_merging

base_path = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files")


suffixes = {
    "toe_loaded_tension_strain" : "_toe_loaded_tension_strain.csv",
    "toe_loaded_thickness" : "_toe_loaded_thicknesses.csv",
    "max_pressure" : "_max_pressure.csv",
    "relaynet_thicknesses" : "_thickness.csv",
    "matlab_apex_rise_pressure" : "_Pressure_Apex.csv",
    "matlab_raw_apex_rise" : "_Apex_raw.csv"
    }

#%% FIND PATHS TO FILES

dict_path_files = {}

list_path_all_files_dirs = list(base_path.rglob("*"))
list_path_all_files = [p for p in list_path_all_files_dirs if p.is_file()]

# get pressure apex files 

list_apex_rise_pressure_file = list(base_path.rglob(f"*{suffixes['matlab_apex_rise_pressure']}"))

# iterate through the pressure files 
for path_apex_pressure_file in list_apex_rise_pressure_file[:]:
    pass
    base_name = path_apex_pressure_file.stem.rsplit("_",2)[0]

    # initialize dict for this sample    
    dict_path_files[base_name] = {}
    
    # find all other files by generating the suffix
    for file_suffix in suffixes:
        pass
        # pressure files look at all sample images so they are missing e.g. "_0001_Mode2D"
        if file_suffix == "max_pressure":
            filename =  f"{base_name.rsplit('_',2)[0]}{suffixes[file_suffix]}"
        else:
            filename = f"{base_name}{suffixes[file_suffix]}"
        
        path_file = [p for p in list_path_all_files if filename in str(p)]
        
        if len(path_file) > 1:
            print("more than one file found!!!!!")
            print(path_file)
        elif len(path_file) != 0:
            dict_path_files[base_name][file_suffix] = path_file[0]
        else:
            dict_path_files[base_name][file_suffix] = ""


#%% seprate complete sets

dict_complete_sets = {}

for sample_key in dict_path_files:
    pass
    sample_set = dict_path_files[sample_key]
    if sample_set["toe_loaded_tension_strain"] != "" and sample_set["toe_loaded_tension_strain"].exists() and \
        sample_set["max_pressure"] != "" and sample_set["max_pressure"].exists() and \
        sample_set["matlab_apex_rise_pressure"] != "" and sample_set["matlab_apex_rise_pressure"].exists():

        # only amniochorion will have these 
        # sample_set["relaynet_thicknesses"] != "" and sample_set["relaynet_thicknesses"].exists() and \
        # sample_set["toe_loaded_thickness"] != "" and sample_set["toe_loaded_thickness"].exists() and \

        # save set
        dict_complete_sets[sample_key] = sample_set
    


#%% load complete sets and merge


# iterate through complete sets, merge and save into corresponding folders 
for complet_set_key in tqdm(list(dict_complete_sets)[:]):
    pass

    # clear vars
    df_matlab_apex_rise_pressure = ""
    df_relaynet_thicknesses = ""
    df_toe_loaded_thickness = ""
    df_max_pressure = ""
    df_toe_loaded_tension_strain = ""
    df_matlab_raw_apex_rise = ""

    path_output = dict_complete_sets[complet_set_key]["matlab_apex_rise_pressure"].parent
    base_filename = dict_complete_sets[complet_set_key]["matlab_apex_rise_pressure"].stem.rsplit("_",2)[0]
    
    # LOAD DATA INTO DATAFRAMES
    df_matlab_apex_rise_pressure = pd.read_csv(dict_complete_sets[complet_set_key]["matlab_apex_rise_pressure"],
                                               names=["apex_rise","pressure"])
    
    # ADD THICKNESS TO FILES IF IT EXISTS 
    if dict_complete_sets[complet_set_key]["relaynet_thicknesses"] != "" and (path_thick := dict_complete_sets[complet_set_key]["relaynet_thicknesses"]).exists():
        pass
        df_relaynet_thicknesses = pd.read_csv(path_thick)
        df_toe_loaded_thickness = pd.read_csv(dict_complete_sets[complet_set_key]["toe_loaded_thickness"])
        path_thick_exists = True
    else:
        path_thick_exists = False
        


    df_max_pressure = pd.read_csv(dict_complete_sets[complet_set_key]["max_pressure"])

    ## these have all params from the grid search, pick which to keep
    df_toe_loaded_tension_strain = pd.read_csv(dict_complete_sets[complet_set_key]["toe_loaded_tension_strain"])
    
    # merge raw matlab export if available
    if bool_has_raw_matlab_export := (dict_complete_sets[complet_set_key]["matlab_raw_apex_rise"] != ""):
        df_matlab_raw_apex_rise = pd.read_csv(dict_complete_sets[complet_set_key]["matlab_raw_apex_rise"], \
                                              names=["raw_apex_rise"])

    # START MERGING
  
    ###################### TENSION STRAIN
    # df_toe_loaded_tension_strain
    # TODO specify which value to same from parameter search
    toe_loaded_threshold_row_to_keep = "toe_0.5-5_loaded_7.5-17.8"
    row_toe_loaded_tension_strain = df_toe_loaded_tension_strain[df_toe_loaded_tension_strain["sample_name"] == toe_loaded_threshold_row_to_keep]
    row_toe_loaded_tension_strain = row_toe_loaded_tension_strain.reset_index(drop=True)
    
    def _string_to_list(list_as_str : str) -> list:
        pass
        str_list_no_brackets = list_as_str[1:-1]
        return [float(value.strip("\n")) for value in str_list_no_brackets.split(" ") if value != ""] # list of values
    
    # insert tension and strain
    tension = _string_to_list(row_toe_loaded_tension_strain["tension"][0])
    strain = _string_to_list(row_toe_loaded_tension_strain["strain"][0])
    indices = np.asarray(_string_to_list(row_toe_loaded_tension_strain["frame_indices"][0]), dtype=int) -1 # zero indexed
    
    # initialize columns 
    df_merged = pd.DataFrame()
    df_merged["tension"] = ["NaN"] * len(tension)
    df_merged["strain"] = ["NaN"] * len(strain)

    for idx,val_tension, val_strain in zip(indices,tension,strain):
        pass
        df_merged.at[idx, "tension"] = val_tension
        df_merged.at[idx, "strain"] = val_strain
        
    df_merged["threshold_toe_low"] = [row_toe_loaded_tension_strain["threshold_toe_low"][0]] * len(df_merged)
    df_merged["threshold_toe_high"] = [row_toe_loaded_tension_strain["threshold_toe_high"][0]] * len(df_merged)
    df_merged["threshold_loaded_low"] = [row_toe_loaded_tension_strain["threshold_loaded_low"][0]] * len(df_merged)
    df_merged["threshold_loaded_high"] = [row_toe_loaded_tension_strain["threshold_loaded_high"][0]] * len(df_merged)
    df_merged["max_tension"] = [row_toe_loaded_tension_strain["max_tension"][0]] * len(df_merged)
    df_merged["max_strain"] = [row_toe_loaded_tension_strain["max_strain"][0]] * len(df_merged)
    df_merged["max_apex"] = [row_toe_loaded_tension_strain["max_apex"][0]] * len(df_merged)
    df_merged["toe_modulus"] = [row_toe_loaded_tension_strain["toe_modulus"][0]] * len(df_merged)
    df_merged["loaded_modulus"] = [row_toe_loaded_tension_strain["loaded_modulus"][0]] * len(df_merged)
    
    
    ###################### APEX RISE AND THICKNESS MERGING
    ### check that they are the same length, pad df_thickness if necessary
    if path_thick_exists and not len(df_matlab_apex_rise_pressure) == len(df_relaynet_thicknesses):
        print("padding thickness file")
        df_relaynet_thicknesses = pad_relaynet_for_merging(df_matlab_apex_rise_pressure, df_relaynet_thicknesses)
        df_merged = pd.concat([df_merged, df_matlab_apex_rise_pressure, df_relaynet_thicknesses], axis=1)
    elif path_thick_exists:
        df_merged = pd.concat([df_merged, df_matlab_apex_rise_pressure, df_relaynet_thicknesses], axis=1)
    else: #just pad pressure_apex_rise
        df_merged = pd.concat([df_merged, df_matlab_apex_rise_pressure], axis=1)


    # df_max_pressure
    df_merged["location"] = [df_max_pressure["location"][0]] * len(df_merged)
    df_merged["layers"] = [df_max_pressure["layers"][0]] * len(df_merged)
    df_merged["birth_type"] = [df_max_pressure["birth_type"][0]] * len(df_merged)
    df_merged["max_pressure"] = [df_max_pressure["max_pressure"][0]] * len(df_merged)
    
    ###################### RELAYNET THICKNESSES

    # specific to ReLaYNet exports
    if path_thick_exists:
        # df_toe_loaded_thickness
        row_toe_loaded_thickness = df_toe_loaded_thickness[df_toe_loaded_thickness["sample_name"] == toe_loaded_threshold_row_to_keep]
        row_toe_loaded_thickness = df_toe_loaded_thickness[df_toe_loaded_thickness["sample_name"] == toe_loaded_threshold_row_to_keep]
        row_toe_loaded_thickness = row_toe_loaded_thickness.reset_index(drop=True)
    
        df_merged['avg_amnion_toe_thickness'] =  [row_toe_loaded_thickness["avg_amnion_toe_thickness"][0]] * len(df_merged)
        df_merged['avg_spongy_toe_thickness'] =  [row_toe_loaded_thickness["avg_spongy_toe_thickness"][0]] * len(df_merged) 
        df_merged['avg_chorion_toe_thickness'] =  [row_toe_loaded_thickness["avg_chorion_toe_thickness"][0]] * len(df_merged)
        df_merged['avg_decidua_toe_thickness'] =  [row_toe_loaded_thickness["avg_decidua_toe_thickness"][0]] * len(df_merged)
        df_merged['avg_amnion_spongy_chorion_toe_thickness'] =  [row_toe_loaded_thickness["avg_amnion_spongy_chorion_toe_thickness"][0]] * len(df_merged)
        df_merged['avg_amnion_loaded_thickness'] =  [row_toe_loaded_thickness["avg_amnion_loaded_thickness"][0]] * len(df_merged)
        df_merged['avg_spongy_loaded_thickness'] =  [row_toe_loaded_thickness["avg_spongy_loaded_thickness"][0]] * len(df_merged)
        df_merged['avg_chorion_loaded_thickness'] =  [row_toe_loaded_thickness["avg_chorion_loaded_thickness"][0]] * len(df_merged)
        df_merged['avg_decidua_loaded_thickness'] =  [row_toe_loaded_thickness["avg_decidua_loaded_thickness"][0]] * len(df_merged)
        df_merged['avg_amnion_spongy_chorion_loaded_thickness'] =  [row_toe_loaded_thickness["avg_amnion_spongy_chorion_loaded_thickness"][0]] * len(df_merged)
            
    ###################### RAW APEX RISE, if available 
    if bool_has_raw_matlab_export:
        df_merged = pd.concat([df_merged, df_matlab_raw_apex_rise], axis=1)
    
    ###################### assert layers, location and birth_type match
    if path_thick_exists:
        assert row_toe_loaded_thickness["location"][0] == row_toe_loaded_tension_strain["location"][0] == df_max_pressure["location"][0], "location mismatch on files"
        assert row_toe_loaded_thickness["birth_type"][0] == row_toe_loaded_tension_strain["birth_type"][0] == df_max_pressure["birth_type"][0], "birth_type mismatch on files"
        assert row_toe_loaded_thickness["layers"][0] == row_toe_loaded_tension_strain["layers"][0] == df_max_pressure["layers"][0], "layers mismatch on files"
    else:
        assert row_toe_loaded_tension_strain["location"][0] == df_max_pressure["location"][0], "location mismatch on files"
        assert row_toe_loaded_tension_strain["birth_type"][0] == df_max_pressure["birth_type"][0], "birth_type mismatch on files"
        assert row_toe_loaded_tension_strain["layers"][0] == df_max_pressure["layers"][0], "layers mismatch on files"
    

    # reindex rowsto star at 1
    # df_merged.index = np.arange(1, len(df_relaynet_thicknesses)+1)
    # df_merged.index.names = ["frame_number"] 
    
    path_features_csv = path_output / f"{base_filename}_features.csv"
    df_merged.index.name = "index"
    df_merged.to_csv(path_features_csv, na_rep="NA")
    print(path_features_csv.name)