from pathlib import Path
import pandas as pd
from tqdm import tqdm

path_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files")

path_summaries = path_dataset.parent / "data_summaries"

list_path_features_csv = list(path_dataset.rglob("*_features.csv"))

import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
#%% header 

dict_data_common_to_samples = {
#common to amniochorion and amnion and chorion

    # experiment identifiers 
    # "base_name" : ""
    "location" :'location',
    "layers" : 'layers',
    "birth_type" : 'birth_type',
    
    # apex rise and pressure 
    "max_apex" : 'max_apex',
    "max_strain" : 'max_strain',
    "max_pressure" :  'max_pressure',

    # tension/strain script
    "tension" : 'tension',
    "strain" : 'strain',
    'threshold_toe_low' : 'threshold_toe_low', 
    'threshold_toe_high' : 'threshold_toe_high',
    'threshold_loaded_low' : 'threshold_loaded_low',
    'threshold_loaded_high' : 'threshold_loaded_high',
    "toe_modulus" : 'toe_modulus',
    "loaded_modulus" : 'loaded_modulus',

}
    #############
            #frame by frame values 
           #  'apex_rise', 'pressure',
           # 'decidua-thickness', 'decidua-area', 'decidua-length',
           # 'chorion-thickness', 'chorion-area', 'chorion-length',
           # 'spongy-thickness', 'spongy-area', 'spongy-length', 'amnion-thickness',
           # 'amnion-area', 'amnion-length', 'total_thickness', 
           ## specific to amniochorion 

# thickness data for toe and loaded regions
data_only_amniochorion = {

    # toe 
    'avg_amnion_toe_thickness' : 'avg_amnion_toe_thickness',
    'avg_spongy_toe_thickness' : 'avg_spongy_toe_thickness', 
    'avg_chorion_toe_thickness' : 'avg_chorion_toe_thickness',
    'avg_decidua_toe_thickness' : 'avg_decidua_toe_thickness', 
    'avg_amnion_spongy_chorion_toe_thickness' : 'avg_amnion_spongy_chorion_toe_thickness',
    
    # loaded 
    'avg_amnion_loaded_thickness' : 'avg_amnion_loaded_thickness', 
    'avg_spongy_loaded_thickness' : 'avg_spongy_loaded_thickness',
    'avg_chorion_loaded_thickness' : 'avg_chorion_loaded_thickness', 
    'avg_decidua_loaded_thickness' : 'avg_decidua_loaded_thickness',
    'avg_amnion_spongy_chorion_loaded_thickness' : 'avg_amnion_spongy_chorion_loaded_thickness'  
    }


#%%
df_all_features = pd.DataFrame()

dict_summary = {}

for path_features  in tqdm(list_path_features_csv):
    pass
    
    # load this features.csv sheet 
    df_temp = pd.read_csv(path_features, index_col="index")    
    sample_name  = str(path_features.stem).rsplit("_",1)[0]
    
    dict_summary[sample_name] = {}
    
    # populate common entries
    for common_entry in dict_data_common_to_samples:
        pass
        # grab values from first entry
        dict_summary[sample_name][common_entry] = list(df_temp.iloc(0))[0][common_entry]

    # populate amniochorion entries only
    if dict_summary[sample_name]["layers"] == "amniochorion":
        for entry in data_only_amniochorion:
            pass
            dict_summary[sample_name][entry] = list(df_temp.iloc(0))[0][entry]



df_summary = pd.DataFrame(dict_summary)

df_summary = df_summary.transpose()

df_summary.index.name = "sample_name"

df_summary.to_csv(path_summaries / "all_samples_data_summary.csv")

    # df_temp["sample_name"] = str(path_features.stem).rsplit("_",1)[0]

    # if len(df_all_features) == 0:
    #     df_all_features = df_temp
    # else:
    #     df_all_features = pd.concat([df_all_features,df_temp], axis=0)
        
df_amniochorion = df_summary[df_summary["layers"] == "amniochorion"]

#%% quick plots 

import holoviews as hv
hv.extensions('bokeh')
from holoviews import opts

path_figures = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\figures")


    
    
