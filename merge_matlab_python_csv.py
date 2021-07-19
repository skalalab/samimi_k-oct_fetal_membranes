from pathlib import Path
import pandas as pd


path_sample = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\oct_dataset_3100x256\0-segmentation_completed\2018_10_09_human_amniochorion_labored_term_AROM_pericervical_0002_Mode2D".replace("\\",'/'))


# load matlab export csv
path_matlab_csv = list(path_sample.glob("*_Pressure_Apex.csv"))[0]
pd_frame_apex_rise_pressure = pd.read_csv(path_matlab_csv, names=["Apex Rise", "Pressure"])

# load python thickness
path_python_csv = Path(r"Z:\0NDL8U~X\K5VHRD~Q\O7TBB9~W\0V1FKE~0\23OQ4F~N\TJ88EI~L\2AHKSN~O.CSV".replace("\\", '/'))
pd_frame_thickness = pd.read_csv(path_python_csv)

# merge dataframes
merged_df = pd.concat([pd_frame_apex_rise_pressure, pd_frame_thickness], axis=1)

merged_df.to_csv(path_sample / f"{path_sample.name}_feats.csv")
