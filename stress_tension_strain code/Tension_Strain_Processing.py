from pathlib import Path
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from numpy.linalg import lstsq
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300



path_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files")
                
list_path_features_csv = list(path_dataset.rglob("*amniochorion*features.csv"))


# path_csv = Path(r"Z:\0NDL8U~X\K5VHRD~Q\HS875H~W\2S9W4P~Z\IIZ6KD~H\2CYKSR~Z\2OPO3V~E.CSV")


df = pd.read_csv(path_csv)
df = df.dropna()
df_pressure_apex = df[["frame_number","Apex Rise", "Pressure"]]

#%% Extract indices for toe and loading region

# get indices from pressure array 
def get_region_indices(data, val_lower, val_upper):
    # extract regions based on first instance of lower and upper bounds
    idx_lower = np.argwhere(data > val_lower).squeeze()[0]
    idx_upper = np.argwhere(data > val_upper).squeeze()
    if idx_upper.size == 0: # no values larger than upper, find idx of max value 
        idx_upper = np.argwhere(data == data.max()).squeeze()[0]
    else:
        idx_upper = idx_upper[0]
    return idx_lower, idx_upper

# toe range
thresh_toe_low = 0.5
thresh_toe_high = 5
idx_toe = get_region_indices(df_pressure_apex["Pressure"].values,
                             thresh_toe_low,
                             thresh_toe_high)
# loaded region range 
thresh_loaded_low = 7.5
thresh_loaded_high = np.max(df_pressure_apex["Pressure"].values) # or 17.8
idx_loaded = get_region_indices(df_pressure_apex["Pressure"].values,
                             thresh_loaded_low,
                             thresh_loaded_high)

## plot regions
plt.title("Apex Rise vs Pressure")
plt.plot(df_pressure_apex["Apex Rise"],df_pressure_apex["Pressure"])
#toe region
plt.plot(df_pressure_apex["Apex Rise"][idx_toe[0]:idx_toe[1]], 
         df_pressure_apex["Pressure"][idx_toe[0]:idx_toe[1]], label="toe region")
# loaded region
plt.plot(df_pressure_apex["Apex Rise"][idx_loaded[0]:idx_loaded[1]],
         df_pressure_apex["Pressure"][idx_loaded[0]:idx_loaded[1]] , label="loaded region")
plt.legend()
plt.xlabel("Apex Rise [mm]")
plt.ylabel("Pressure [kPa]")
plt.show()

#%% Tension-Strain Analysis

initial_apex = 6   # in [mm]
initial_pressure = 0 # kPa
device_radius = 15 # radius of  device that holds the membrane
meters_to_mm = 1000

initial_radius = (initial_apex**2 + device_radius**2)/(2 * initial_apex)   # in [mm]

# add offset initial offset to apex and set starting value to initial offset
apex = initial_apex + np.asarray(df_pressure_apex["Apex Rise"]) # add offset
apex = np.insert(apex, 0, initial_apex)

# set initial pressure to zero
pressure = np.asarray(df_pressure_apex["Pressure"])
pressure = np.insert(pressure,0, initial_pressure)

# m ==>mm /2
radius = (apex**2 + device_radius**2) / (2 * apex)   # in [mm]  *****
tension = (pressure * radius) / (2 * meters_to_mm);   # in [N/mm]
strain = apex / radius # *****
#%%
# C_toe = cat(2, Strain(ind_toe_low:ind_toe_high), ones(ind_toe_high-ind_toe_low+1,1))
# d_toe = Tension(ind_toe_low:ind_toe_high);
# lin_coeffs_toe = C_toe\d_toe;                       % solve for linear fit coefficients
# TensionMod_toe = lin_coeffs_toe(1);                 % tangent modulus of the toe region in [N/m]

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

plt.title("Tension vs Strain")
tension_meters = tension*1000
plt.plot(strain, tension_meters)

# TOE REGION

toe_strain = strain[idx_toe[0]:idx_toe[1]]
toe_tension = tension_meters[idx_toe[0]:idx_toe[1]]
#
plt.plot(toe_strain, toe_tension , color="r")
toe_slope, toe_y_int = best_fit_line(toe_strain, toe_tension)
x = toe_strain
plt.plot(x, toe_slope*x + toe_y_int, label="toe", color="r")
plt.text(np.min(toe_strain), np.mean(toe_tension), f"Tension Modulus\n={int(toe_slope)} [N/m] ",color="r")


# LOADED
loaded_strain = strain[idx_loaded[0]:idx_loaded[1]]
loaded_tension = tension_meters[idx_loaded[0]:idx_loaded[1]]

plt.plot(loaded_strain, loaded_tension , color="g")
loaded_slope, loaded_y_int = best_fit_line(loaded_strain, loaded_tension)
x = loaded_strain
plt.plot(x, loaded_slope*x + loaded_y_int, label="loaded", color="g")
plt.text(np.min(loaded_strain)*.9, np.mean(loaded_tension), f"Tension Modulus\n={int(loaded_slope)} [N/m] ",color="g")


## FINISH PLOTTING
plt.legend()
plt.xlabel("Strain")
plt.ylabel("Tension N/m")
plt.show()

#%%

# loaded region


# Tension instead of Stress
# C_loaded = cat(2, Strain(ind_loaded_low:ind_loaded_high), ones(ind_loaded_high-ind_loaded_low+1,1));
# d_loaded = Tension(ind_loaded_low:ind_loaded_high);
# lin_coeffs_loaded = C_loaded\d_loaded;              % solve for linear fit coefficients
# TensionMod_loaded = lin_coeffs_loaded(1);           % tangent modulus of the under-load region in [N/m]




# stress = [strain 1]*[a b]




