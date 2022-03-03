from pathlib import Path
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
import tifffile
from tqdm import tqdm
import numpy as np
from natsort import natsorted
from pprint import pprint

#%% Get list of folder to process

path_dataset = Path(r"Z:\Kayvan\Human Data")

## get all files 
path_all_files = list(path_dataset.rglob("*"))

## get just folders 
path_all_dirs = [p for p in path_all_files if p.is_dir()]

## get just 2p folders 
path_all_dirs_2p = [p for p in path_all_dirs if "\\2p\\" in str(p).lower()]

# get cervical or placental samples

path_samples = [p for p in path_all_dirs_2p if ("cervical" in str(p).lower() or "placental" in str(p).lower()) ]

# get placental or cervical dirs
path_subset_samples = set([])

for dir_sample in path_samples: 
    pass
    path_root = str(dir_sample).split("\\", 6)
    path_root_joined = "/".join(path_root[:6])
    path_subset_samples.add(Path(path_root_joined))    

#%% For each directory, find the folder with most items/images

for dir_sample in tqdm(path_subset_samples):
    pass
    print(dir_sample.parent.parent.name)

    # get folders in dir
    list_subfolders = [p for p in list(dir_sample.glob("*")) if p.is_dir()]
    
    path_dir_most_files = None
    
    # iterate through subdirs
    for path_subfolder in list_subfolders:
        pass
        # initialize subfolder 
        if path_dir_most_files is None:
            path_dir_most_files = path_subfolder
          
        # compare numbe of files
        elif len(list(path_subfolder.glob("*"))) > len(list(path_dir_most_files.glob("*"))):
            path_dir_most_files = path_subfolder
        
    # here we have dir with most files 
    # print(f"most files dir:{path_dir_most_files} ")
    list_images = list(path_dir_most_files.glob("*_Ch2_*"))
    
    # sorted nartually 
    list_images = natsorted(list_images)
    # pprint(list_images)
    
    # for path_im in tqdm(list_images):
    #     im = tifffile.imread(path_im)
    #     plt.imshow(im)
    #     plt.show()
    
    path_output = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\2p")
    
    location = ""
    if "cervical" in path_subfolder.parent.name.lower():
        location = "cervical"
    elif "placental" in path_subfolder.parent.name.lower():
        location = "placental"
    
    filename = f"{path_subfolder.parent.parent.parent.name}_{location}.tiff"
    with tifffile.TiffWriter(path_output / filename, bigtiff=True) as tif:  # imagej=True
        for pos, path_im in enumerate(list_images):
            print(f"saving:{pos}")
            im = tifffile.imread(path_im)
            tif.save(im.astype(np.uint16))
            
