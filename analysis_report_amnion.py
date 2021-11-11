
from pprint import pprint

from pathlib import Path

path_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files")


# get pressure files
layer = "amnion"
list_path_pressure_files = list(path_dataset.rglob(f"*_{layer}_*.txt")) + \
                                list(path_dataset.rglob(f"_{layer}_*Mode2D*.txt"))

for idx, path in enumerate(list_path_pressure_files):
    pass
    
    if "ressure" in str(path):
        base_name = path.stem.rsplit("_", 1)[0]
        path_im = path.parent / f"{base_name}.tiff"
    else:
        #one of the pressure files with Mode2D.txt
        
    # is there a corresponding tiff next to it


    # is there a _mask.tiff for segmented mask?
    # is there a _Pressure_Apex.csv file?