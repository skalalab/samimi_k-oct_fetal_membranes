#%% merge matlab outputs and python into one csv file

from pathlib import Path
import pandas as pd

path_sample = path_image.parent

# path_sample = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\oct_dataset_3100x256\0-segmentation_completed\2018_10_09_human_amniochorion_labored_term_AROM_pericervical_0002_Mode2D".replace("\\",'/'))
# load matlab export csv
path_matlab_csv = list(path_sample.glob("*_Apex.csv"))[0]
df_frame_apex_rise_pressure = pd.read_csv(path_matlab_csv, names=["Apex Rise", "Pressure"])

# load python thickness
path_frame_vs_thickness = path_sample / "thickness"
path_frame_vs_thickness_csv = list(path_frame_vs_thickness.glob("*.csv"))[0]
df_frame_thickness = pd.read_csv(path_frame_vs_thickness_csv)


### check that they are the same length, extend if necessary
if len(df_frame_apex_rise_pressure) != len(df_frame_thickness):
    # this is used in the event that not all frames are segmented by relaynet
    # if for some reason the membrane goes out of screen and segmentation fails
    print("Error: matlab and python export don't match in number of frames")
    print(f"apex df: {len(df_frame_apex_rise_pressure)} | thickness df: {len(df_frame_thickness)}")
    answer = input("Extend thickness df to match apex rise df? (y/n)")
    if answer == "y":
        ## fill thickness df to match apex rise
        num_missing_rows= len(df_frame_apex_rise_pressure) - len(df_frame_thickness)
        df_copy = df_frame_thickness.copy()
        # create list of indices to match apex rise df
        list_new_indices = np.arange(num_missing_rows) + len(df_frame_thickness)
        # print(df_copy.loc[len(df_copy.index)-1])
        filler_row = ["NA"] * len(df_frame_thickness.columns) # create filler row matching number of cols 
        for idx in list_new_indices: #fill indicex with filler row
            df_copy.loc[idx] = filler_row
        df_frame_thickness = df_copy # replace df
    else:
        assert len(df_frame_apex_rise_pressure) == len(df_frame_thickness), "df_apex is not the same length as df_thickness"
# merge dataframes
df_merged = pd.concat([df_frame_apex_rise_pressure, df_frame_thickness], axis=1)

# reindex rowsto star at 1
df_merged.index = np.arange(1, len(df_frame_thickness)+1)
df_merged.index.names = ["frame_number"] 

path_features_csv = path_sample / f"{path_sample.name}_features.csv"
df_merged.to_csv(path_features_csv, na_rep="NA")
print(path_features_csv)