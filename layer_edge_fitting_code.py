import tifffile
from pathlib import Path
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from skimage.morphology import dilation, remove_small_objects
from skimage.measure import label, regionprops

# https://pypi.org/project/hampel/
# https://dsp.stackexchange.com/questions/26552/what-is-a-hampel-filter-and-how-does-it-work
from hampel import hampel

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor
from sklearn.linear_model import RANSACRegressor
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
#%%
########### helper functions


## calculate layer length 2nd method
# placenta is  vertical here, meaning edges are at the top and bottom
# layers increase left to right
def _find_layer_vertices(mask):
    list_layer_vertices = []
    for pos_row, row in enumerate(mask): # iterate through rows 
        for pos_col, pixel_value in enumerate(row): #iterate through cols in row
        
            if (pos_col+1) == len(row): # check if we've reached the end of the array
                break # goto next row
            
            if pixel_value == True: # mask is at edge of mask
                list_layer_vertices.append((pos_row, pos_col)) # save layer pixel
                break # go to next row
            
            if pixel_value != row[pos_col+1]: # next pixel is a layer pixel if different
                list_layer_vertices.append((pos_row, pos_col+1)) # save layer pixel
                break # go to next row
    return list_layer_vertices

###########


# Algorithm --> Smooth curved surfaces
# for each mask -->
# to find the top and bottom
# 	for each column find top and bottom masks and take average


working_dir = Path("C:/Users/Nabiki/Desktop/split_masks_selected")
im_paths = list(working_dir.glob("*.tiff"))


for im_path in im_paths[2:3]:
    mask = tifffile.imread(im_path)
    # plt.imshow(mask)
    print(np.unique(mask))
    plt.show()

    mask = mask.transpose() # image must be vertical and binary
    plt.title("initial working image")
    plt.imshow(mask[800:2300,300:])
    plt.show()
        
    ############ remove small blobs (anything 10% or smaller)
    label_image = label(mask)
    
    label_region_props = regionprops(label_image)
    for pos, region in enumerate(label_region_props):
        # print(f"pos: {pos} area: {region.area}")
        if pos == 0:   # initialize with first region
            largest_region = region  # initialize largest region
        elif region.area > largest_region.area:
            largest_region = region
    layer_mask = remove_small_objects(label_image, min_size=(largest_region.area*0.1))  # remove spots < 10% largest region found
    
    
    layer_mask[layer_mask > 0] = 1
    
    # plt.title("after removing small blobs")
    # plt.imshow(layer_mask)
    # plt.show()
    
    plt.title("after removing small blobs")
    plt.imshow(layer_mask[800:2300,300:])
    plt.show()
    
    ##### store vertices 
    vertices = _find_layer_vertices(layer_mask)
    
    # list_heights, list_index = [height for _, height in vertices] # extract height value
    
    list_rows = [rows for rows, _ in vertices]# extract height value
    list_heights = [cols for _, cols in vertices]
    
    # testing: add outliers
    # list_heights[600] = 480
    # list_heights[200] = 400
    # list_heights[1000] = 500
    # list_heights[1001] = 505
    
    plt.title("plot of vertices")
    plt.plot(list_heights)
    plt.show()
    
    ###### apply hampel filter to remove outliers
    series_heights = pd.Series(list_heights)    
    series_heights_cleaned = hampel(series_heights, window_size=11, n=1) # was 3
    
    plt.title("red: original, black: after hampel filter")
    series_heights.plot(style="r-", alpha=1)
    series_heights_cleaned.plot(style="k-", alpha=1)
    plt.show()
    
    ####### Robtus Fitting RANSAC
    #https://stackoverflow.com/questions/55682156/iteratively-fitting-polynomial-curve/55787598

    data_heights = series_heights_cleaned.values[:,np.newaxis]
    
    ###################
    from sklearn.metrics import mean_squared_error
    class PolynomialRegression(object):
        def __init__(self, degree=2, coeffs=None):
            self.degree = degree
            self.coeffs = coeffs
    
        def fit(self, X, y):
            self.coeffs = np.polyfit(X.ravel(), y, self.degree)
    
        def get_params(self, deep=False):
            return {'coeffs': self.coeffs}
    
        def set_params(self, coeffs=None, random_state=None):
            self.coeffs = coeffs
    
        def predict(self, X):
            poly_eqn = np.poly1d(self.coeffs)
            y_hat = poly_eqn(X.ravel())
            return y_hat
    
        def score(self, X, y):
            return mean_squared_error(y, self.predict(X))
    
    ##########
    poly_degree = 2
    x_vals = list_rows
    y_vals = data_heights.squeeze().tolist()
    
    ransac = RANSACRegressor(PolynomialRegression(degree=poly_degree),
                             residual_threshold=3 * np.std(y_vals),
                             random_state=0)
    ransac.fit(np.expand_dims(x_vals, axis=1), y_vals)
    inlier_mask = ransac.inlier_mask_
    
    
    y_hat = ransac.predict(np.expand_dims(x_vals, axis=1))
    plt.plot(x_vals, y_vals, 'bx', label='input samples')
    plt.plot(np.asarray(x_vals)[inlier_mask], np.asarray(y_vals)[inlier_mask], 'go', label='inliers (3*STD)', markersize=2)
    plt.plot(x_vals, y_hat, 'r-', label='estimated curve')
    plt.legend()


    
    # # ( volume of bead * density(kg/m^3) * area(m^2) )/ per 10ms
    # volume_sphere = (4/3)*np.pi*(5e-6)**3 # m^3
    # density = 1060 # kg/m^3
    # flow_speed = 1e-3 * 100 # 100ms
    # mass_flow_rate = (volume_sphere * density)/flow_speed


# Fit 2nd order polynomial -->
# robust linearly squares --> tolerante to noise
# weighted least squares
# calculate other thick layers and leftover is the amnion
# fit parabola to the top > robust fitting method > 
# ransac --> outliers in the y direction 
# apply hampel filter on the points to smooth out the curve --> 5 or 9 windows size

# find pixel indices --> hampel filter -- then fit --> repeat for bottom curve

# ---- find top and bottom
# --> spline fitting

# difference of gaussians - DOG - find blobs of certain sizes--> convolve --> 

#%%