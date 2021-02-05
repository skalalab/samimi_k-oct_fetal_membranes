import tifffile
from pathlib import Path
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from skimage.morphology import  remove_small_objects
from skimage.measure import label, regionprops

# https://pypi.org/project/hampel/
# https://dsp.stackexchange.com/questions/26552/what-is-a-hampel-filter-and-how-does-it-work
from hampel import hampel

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor
from sklearn.linear_model import RANSACRegressor

#%%
########### helper functions


## calculate layer length 2nd method
# placenta image is vertical here, meaning edges are at the top and bottom
# layers increase left to right
def _find_bottom_layer_vertices(mask):
    list_layer_vertices = []
    n_rows, n_cols = mask.shape # subtracting cols length inverts graph
    for pos_row, row in enumerate(mask): # iterate through rows 
        for pos_col, pixel_value in enumerate(row): #iterate through cols in row
            
            if pixel_value == True: # mask is at edge of mask
                list_layer_vertices.append((pos_row, n_cols-pos_col)) # save layer pixel
                break # go to next row
            
    return list_layer_vertices

## calculate layer length 2nd method
# placenta image is vertical here, meaning edges are at the top and bottom
# layers increase left to right
def _find_top_layer_vertices(mask):
    list_layer_vertices = []
    for pos_row, row in enumerate(mask): # iterate through rows 
        for pos_col, pixel_value in enumerate(row): #iterate through cols in row
        
            if (pos_col+1) == len(row): # check if we've reached the end of the array
                break # goto next row
            
            if pixel_value == True: # mask edge found
                list_layer_vertices.append((pos_row, pos_col)) # save layer pixel
                break # go to next row
    return list_layer_vertices

# Algorithm --> Smooth curved surfaces
# for each mask -->
# to find the top and bottom
# 	for each column find top and bottom masks and take average


working_dir = Path("C:/Users/Nabiki/Desktop/split_masks_selected")
im_paths = list(working_dir.glob("*.tiff"))


#iterate through every mask
for im_path in im_paths: #[3:6]:
    pass

    # LOAD AND SHOW MASK
    mask = tifffile.imread(im_path)
    mask = mask.transpose() ################## image must be vertical and binary
    # plt.title("initial working image")
    # plt.imshow(mask)
    # plt.show()
        
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
    # show mask
    # plt.title("after removing small blobs")
    # plt.imshow(layer_mask)
    # plt.show()
    
    ##### EDGE CALCULATION
    vertices_top = _find_top_layer_vertices(layer_mask)
    vertices_bottom = _find_bottom_layer_vertices(np.flip(layer_mask, axis=1))

    # Show image
    # plt.imshow(np.transpose(mask))
    # plt.show()
    
    list_top_bottom_vertices = [vertices_top, vertices_bottom ]
    
    
    # show edges extracted
    # for vertices in list_top_bottom_vertices:
    #     list_vertex_rows = [rows for rows, _ in vertices] # extract height value
    #     list_vertex_heights = [cols for _, cols in vertices]
    
    #     # show vertices extracted
    #     plt.title("plot of vertices")
    #     plt.scatter(list_vertex_rows, list_vertex_heights, s=1)
    # plt.show()
    
    # Save equation coefficients, min and max
    list_coeffs = []
    
    for vertices in list_top_bottom_vertices:
        pass
    
        ##
        list_vertex_rows = [rows for rows, _ in vertices] # extract height value
        list_vertex_heights = [cols for _, cols in vertices]
    
        ###### apply hampel filter to remove outliers
        series_heights = pd.Series(list_vertex_heights)    
        series_heights_cleaned = hampel(series_heights, window_size=11, n=1) # window size was 3
        
        # plt.title("red: original, black: after hampel filter")
        # series_heights.plot(style="r-", alpha=1)
        # series_heights_cleaned.plot(style="k-", alpha=1)
        # plt.show()
        
        ####### Robtus Fitting RANSAC
        #https://stackoverflow.com/questions/55682156/iteratively-fitting-polynomial-curve/55787598
        
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
        
        x_vals = np.asarray(list_vertex_rows)
        y_vals = np.asarray(series_heights_cleaned)
            
        # plt.title("input to RANSACRegressor ")
        # plt.scatter(x_vals, y_vals, s=1)
        # plt.show()
        
        poly_degree = 2
        stdevs = 2
        residual_threshold = stdevs * np.std(y_vals)
        ransac = RANSACRegressor(PolynomialRegression(degree=poly_degree),
                                  residual_threshold=residual_threshold,
                                  random_state=0)
        
        ransac.fit(np.expand_dims(x_vals, axis=1), y_vals)
        inlier_mask = ransac.inlier_mask_
        
        y_hat = ransac.predict(np.expand_dims(x_vals, axis=1))
        # plt.plot(x_vals, y_vals, 'bx', label='input samples')
        # plt.plot(x_vals[inlier_mask], y_vals[inlier_mask], 'go', markersize=2, label=f'inliers ({str(stdevs)}*STD)')
        # plt.plot(x_vals, y_hat, 'r-', label='estimated curve')
        # plt.legend()
        # plt.show()
        
        list_vertex_rows
        
        coeffs = np.polyfit(x_vals, y_hat, poly_degree)
        list_coeffs.append([coeffs, (np.min(list_vertex_rows), np.max(list_vertex_rows))])
        
    # calculate coeffs
    mask = np.transpose(mask)
    plt.imshow(mask)
    list_layer_lengths = []
    n_rows, n_cols = mask.shape
    for equation_params in list_coeffs:
        pass
        coeffs, (min_value,max_value) = equation_params
        # coeffs = np.polyfit(x_vals, y_hat, poly_degree)
        poly_eqn = np.poly1d(coeffs)
        
        # y values part of the image
        x_vals = np.linspace(0, n_cols, num=(n_cols +1))
        y_hat = poly_eqn(x_vals)
        bool_array_valid_points = y_hat <= n_rows
        
        x_vals_within_image = x_vals[bool_array_valid_points]
        y_vals_within_image = y_hat[bool_array_valid_points]
        
        
        # draw line with valid points
        length = 0
        for pos, (x, y) in enumerate(zip(*[x_vals_within_image, y_vals_within_image])) :
            pass
            index_next_point = pos+1
            
            #last value, no next point
            if (index_next_point) == len(x_vals_within_image):
                break
            # calculate distance
            x_dist = np.absolute(x-x_vals_within_image[index_next_point])
            y_dist = np.absolute(y-y_vals_within_image[index_next_point])
            length +=np.sqrt(x_dist**2 + y_dist**2) # pythagoras        
        list_layer_lengths.append(length)
        
        #plot image
        plt.plot(x_vals_within_image, y_vals_within_image, '-', label='estimated curve', linewidth=1)
    
    #calculate mean and finally plot lines
    mean_length = np.mean(list_layer_lengths)
    print(f"top layer: {list_layer_lengths[0]} bottom layer: {list_layer_lengths[1]} mean: {mean_length}")
    plt.show()
    
    
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