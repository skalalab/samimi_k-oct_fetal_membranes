# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:28:22 2022

@author: econtrerasguzman
"""

from pathlib import Path
import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension("bokeh")

path_dataset_summary = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\data_summaries\all_samples_data_summary.csv")

path_figures = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\figures")

df_dataset_summary = pd.read_csv(path_dataset_summary, index_col="index")
#%%   ############## TOE AND LOADED THICKNESS 

# convert to dataframe
# df_thick = df_dataset_summary[df_dataset_summary["layers"] == "amniochorion"]
# df_thick = df_thick.dropna()

# path_output = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\figures\temp_thickness") # thickness_toe_loaded

# # toe
# kdims = [("term", "Term"),("location","Location")]
   
# bw_toe_amnion = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_amnion_toe_thickness","Avg Thickness (px)")], label="Amnion Toe")
# bw_toe_spongy = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_spongy_toe_thickness","Avg Thickness (px)")], label="Spongy Toe")
# bw_toe_chorion = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_chorion_toe_thickness","Avg Thickness (px)")], label="Chorion Toe")
# bw_toe_decidua = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_decidua_toe_thickness","Avg Thickness (px)")], label="Decidua Toe")
# bw_toe_combined = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_amnion_spongy_chorion_toe_thickness","Avg Thickness (px)")], label="Amion,Spongy,Chorion Toe")

# layout_toe = holoviews_apex_pressure + bw_toe_amnion + bw_toe_spongy + bw_toe_chorion + bw_toe_decidua+  bw_toe_combined

# # LOADED 
# bw_loaded_amnion = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_amnion_loaded_thickness","Avg Thickness (px)")], label="Amnion Loaded")
# bw_loaded_spongy = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_spongy_loaded_thickness","Avg Thickness (px)")], label="Spongy Loaded")
# bw_loaded_chorion = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_chorion_loaded_thickness","Avg Thickness (px)")], label="Chorion Loaded")
# bw_loaded_decidua = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_decidua_loaded_thickness","Avg Thickness (px)")], label="Decidua Loaded")
# bw_loaded_combined = hv.BoxWhisker(df_thick, kdims , vdims=[("avg_amnion_spongy_chorion_loaded_thickness","Avg Thickness (px)")], label="Amnion,Spongy,Chorion Loaded")

# layout_loaded =  overview_holoviews_apex_pressure + bw_loaded_amnion + bw_loaded_spongy + bw_loaded_chorion + bw_loaded_decidua + bw_loaded_combined

# #global options
# overlay = layout_toe + layout_loaded
# overlay.opts(        
#     opts.BoxWhisker(width=500, height=500, tools=["hover"], legend_position='right'),
#     opts.Scatter(width=500, height=500, tools=["hover"], legend_position='top_left', alpha=1),
# ).cols(6)


# str_loaded_range = f"{df_thick['threshold_loaded_low'][0]}_to_{df_thick['threshold_loaded_high'][0]}"
# hv.save(overlay, path_output / f"thickness_loaded_range_{str_loaded_range}.html")

        


#%% ############## TOE AND LOADED TENSIONS TRAIN
# path_output = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\figures\tension_strain_toe_loaded")


# kdims = [("term","Pregnancy Term"),
#           ("location", "Location")
#           ] 
# vdims = [("toe_tension","Toe Tension Modulus")]

 
# # extract toe and loaded values
# toe_low = df_tension_strain['threshold_toe_low'][0]
# tow_high = df_tension_strain['threshold_toe_high'][0]
# loaded_low = df_tension_strain['threshold_loaded_low'][0]
# loaded_high = df_tension_strain['threshold_loaded_high'][0]

# boxwhisker_toe = hv.BoxWhisker(df_loaded_tension_greater_than_toe,kdims, vdims)
# boxwhisker_toe.opts(title=f"Toe Region ({toe_low} to {tow_high})", tools=["hover"])
# # violin_toe = hv.Violin(df_loaded_tension_greater_than_toe, kdims, vdims)
# # plots_toe = boxwhisker_toe * violin_toe

# vdims = [("loaded_tension","Loaded Tension Modulus")]
# boxwhisker_loaded = hv.BoxWhisker(df_loaded_tension_greater_than_toe, kdims, vdims)
# boxwhisker_loaded.opts(title=f"Loaded Region ({loaded_low} to {loaded_high})", tools=["hover"])

# layout = holoviews_apex_pressure + boxwhisker_toe +  overview_holoviews_apex_pressure + boxwhisker_loaded
# layout.opts(
#     opts.BoxWhisker(width=500, height=500),
#     opts.Scatter(width=500, height=500)
#     ).cols(2)

# str_loaded_range = f"{loaded_low}_to_{loaded_high}"
# hv.save(layout, path_output / f"tension_strain_range_{str_loaded_range}_loaded.html")


#%% ############## PEAK PRESSURE ANALYSIS
 

#amniochorion 
df_dropped = df_dataset_summary.dropna() # if you don't do this it won't plot data

boxwhisker = hv.BoxWhisker(df_dataset_summary, ["location", "layers"], "max_pressure", label="Max Pressure kPa" )
boxwhisker.opts(xrotation=90, width=800, height=800, tools=["hover"])

# display counts for each
for layers in ["amnion", "amniochorion", "chorion"]:
    pass
    for loc in ["pericervical", "periplacental"]:
        pass
        df_loc = df_dataset_summary[df_dataset_summary["location"]== loc]
        df_loc_layer = df_loc[df_loc["layers"] == layers]
        print(f"{loc} | {layers}  : {len(df_loc_layer)}")

# export 
hv.save(boxwhisker, path_figures / "apex_rise_pressures_boxwhisker.html")

 