
from pathlib import Path
import tifffile
import matplotlib.pylab as plt
from flim_tools.image_processing import kmeans_threshold
from flim_tools.visualization import compare_images

#%%
dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files")


list_path_amnion_pressure = list(dataset.rglob("*_amnion_*ressure.txt"))  + \
    list(dataset.rglob("*_amnion_*Mode2D.txt"))  


for path_pressure_file in list_path_amnion_pressure[8:9]:
    pass
    base_filename = path_pressure_file.stem.rsplit("_",1)[0]
    path_tiff = (path_pressure_file.parent / f"{base_filename}.tiff")
    if path_tiff.exists():
        print("loading image")
        im = tifffile.imread(path_tiff)
    
    for image in im[50:60]:
        pass
        mask= kmeans_threshold(image[...,0], k=2, n_brightest_clusters=1)
        
        # compare_images(image[...,0], "original", mask, "k means mask")
        plt.imshow(image[...,0])
        plt.show()
        plt.imshow(mask)
        plt.show()
"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files\2020_12_18_C_section_39w0d\Inflation\2020_12_18_C_section_39w0d_pericervical_amnion\2020_12_18_C_section_39w0d_pericervical_amnion_0001_Mode2D_Pressure.txt"