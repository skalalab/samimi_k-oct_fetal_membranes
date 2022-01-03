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
for file in list_feature_csv_files:
    pass
    df = pd.read_csv(file)
    
    df = df.dropna() # drop NA values
    
    fig, ax = plt.subplots(2,2, figsize=(15,12))
    fig.suptitle(file.stem)
    
    # pressure vs apex rise graph
    df_pressure_apex = df[['Apex Rise', 'Pressure']]
    ax[0,0].scatter(df_pressure_apex['Pressure'], df_pressure_apex['Apex Rise'], s=1)
    ax[0,0].set_title("apex rise vs pressure")
    ax[0,0].set_xlabel("pressure [kPa]")
    ax[0,0].set_ylabel("apex displacement [mm]")
    ax[0,0].axis("tight") # equal
    ax[0,0].set_box_aspect(1) # makes plot a square
    
    
    # plot thickness
    df_thickness = df[['decidua-thickness', 'chorion-thickness', 'spongy-thickness', 'amnion-thickness' ]]
    for key_thickness in df_thickness.keys():
        pass
        # ax[0,1].plot(df[key_thickness], label=f"{key_thickness.split('-',1)[0]}")
        ax[0,1].scatter(np.arange(len(df[key_thickness])), df[key_thickness], label=f"{key_thickness.split('-',1)[0]}", s=1)
    ax[0,1].set_title("thickness")
    ax[0,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0,1].set_xlabel("frame")
    ax[0,1].set_ylabel("thickness(pixels)")
    ax[0,1].axis("tight") # equal
    ax[0,1].set_box_aspect(1) # makes plot a square 
    
    # plot areas
    df_area = df[['decidua-area', 'chorion-area', 'spongy-area', 'amnion-area' ]]
    for key_area in df_area:
        ax[1,0].scatter(np.arange(len(df[key_area])), df_area[key_area], label=f"{key_area.split('-',1)[0]}", s=1)
    ax[1,0].set_title("Area")
    ax[1,0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[1,0].set_xlabel("frame")
    ax[1,0].set_ylabel("pixels")
    ax[1,0].axis("tight") # equal
    ax[1,0].set_box_aspect(1) # makes plot a square 
    
    # plot lengths
    df_length = df[['decidua-length', 'chorion-length', 'spongy-length', 'amnion-length' ]]
    for key_length in df_length:    
        ax[1,1].scatter(np.arange(len(df[key_length])), df_length[key_length], label=f"{key_length.split('-',1)[0]}", s=1)
    ax[1,1].set_title("Length")
    ax[1,1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[1,1].set_xlabel("frame")
    ax[1,1].set_ylabel("pixels")
    ax[1,1].axis("tight") # equal
    # https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/axes_box_aspect.html
    ax[1,1].set_box_aspect(1) # makes plot a square 
    
    plt.tight_layout()
    path_figure = file.parent / f"{file.stem.rsplit('_',1)[0]}_fig.png"
    plt.savefig(path_figure)
    print(path_figure)
    plt.show()
    
    # thickness graph


#%%


# from pathlib import Path
# import tifffile
# import numpy as np
# import matplotlib.pylab as plt

# path_dataste = Path(r"Z:\0-Projects and Experiments\TQ - AAG - mitochondria-actin segmentation\mitometer\nadh\mitometer_test")

# list_path_all_images = list(path_dataste.glob("*.tif"))

# im = tifffile.imread(list_path_all_images[0])

# n_rows, n_cols = im.shape
# # _, n_rows, n_cols = im.shape #nadh

# # cube_dataset = np.zeros((n_rows,n_cols, len(list_path_all_images)))
# # for idx, path in enumerate(list_path_all_images):
# #     cube_dataset[:,:,idx] = tifffile.imread(path)


# output_path = Path(r"C:\Users\admin2.ECONTRERAS\Desktop") / f"sw38_nadh_cube.tif"

# with tifffile.TiffWriter(output_path) as tif:
#     for path_im in list_path_all_images:
#         im = tifffile.imread(path_im) # [0,...] #nadh
#         tif.save(im )










