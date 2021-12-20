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


def compute_layer_thickness(mask, method, debug=True):
    
    # validation
    if method not in [1,2]:
        print(f"Error invalid method id: {method}")
        return
    
    
    if debug:
        plt.title("input mask")
        plt.imshow(np.transpose(mask))
        plt.show()
        
    ############ remove small blobs (anything 10% or smaller)
    label_image, num_labels = label(mask, return_num=True)
    label_region_props = regionprops(label_image)
    for pos, region in enumerate(label_region_props):
        # print(f"pos: {pos} area: {region.area}")
        if pos == 0:   # initialize with first region
            largest_region = region  # initialize largest region
        elif region.area > largest_region.area:
            largest_region = region
    
    # remove 
    layer_mask = label_image # this is done to supress warning about removing blobs on binary mask
    if num_labels > 2: # if num labels == 2 you only have bg and one connected component
        layer_mask = remove_small_objects(label_image, min_size=(largest_region.area*0.1))  # remove spots < 10% largest region found
    
    layer_mask[layer_mask > 0] = 1 # make mask binary of area will be off
    
    if debug:
        plt.title("after removing small blobs")
        plt.imshow(layer_mask)
        plt.show()
    
    ##### EDGE CALCULATION
    vertices_top = _find_top_layer_vertices(layer_mask)
    vertices_bottom = _find_bottom_layer_vertices(np.flip(layer_mask, axis=1))


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
        list_heights_cleaned = hampel(series_heights, window_size=11, n=1, imputation=True) # window size was 3
        
        # print(len(list_vertex_rows))
        # print(len(list_vertex_heights))
        # print(len(series_heights))
        # print(len(list_heights_cleaned))
        
        if debug:
            plt.title("red: original, black: after hampel filter")
            series_heights.plot(style="r-", alpha=1)
            series_heights_cleaned = pd.Series(list_heights_cleaned)  
            series_heights_cleaned.plot(style="k-", alpha=1)
            plt.show()
        
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
        y_vals = np.asarray(list_heights_cleaned)
        
        if debug:
            plt.title("input to RANSACRegressor")
            plt.scatter(x_vals, y_vals, s=1)
            plt.show()
        
        poly_degree = 2
        stdevs = 3
        residual_threshold = stdevs * np.std(y_vals)
        ransac = RANSACRegressor(PolynomialRegression(degree=poly_degree),
                                  residual_threshold=residual_threshold,
                                  random_state=0)
        
        ransac.fit(np.expand_dims(x_vals, axis=1), y_vals)
        inlier_mask = ransac.inlier_mask_
        
        y_hat = ransac.predict(np.expand_dims(x_vals, axis=1))
        if debug:
            plt.plot(x_vals, y_vals, 'bx', label='input samples')
            plt.plot(x_vals[inlier_mask], y_vals[inlier_mask], 'go', markersize=2, label=f'inliers ({str(stdevs)}*STD)')
            plt.plot(x_vals, y_hat, 'r-', label='estimated curve')
            plt.legend()
            plt.show()
        
        list_vertex_rows
        
        coeffs = np.polyfit(x_vals, y_hat, poly_degree)
        list_coeffs.append([coeffs, (np.min(list_vertex_rows), np.max(list_vertex_rows))])


    mask = np.transpose(mask) # compute things on a horizontal mask
    n_rows, n_cols = mask.shape
    # plt.imshow(mask)
    # plt.show()

    
    # METHOD 1 - CALCULATE MIDDLE LINE
    if method == 1:
        #print("method 1 selected")
        poly_top_edge = np.poly1d(list_coeffs[0][0])
        poly_bottom_edge = np.poly1d(list_coeffs[1][0])
    
        x_vals = np.linspace(0, n_cols, num=(n_cols +1)) # +1 to produce whole numbers/pixels
        
        y_hat_top = poly_top_edge(x_vals) # top is lower number due to origin on top left
        y_hat_bottom = poly_bottom_edge(x_vals)
        
        y_hat_middle = y_hat_top + ((y_hat_bottom - y_hat_top)/2)
        
        # Fit new line and get y values
        coeffs_middle_poly = np.polyfit(x_vals, y_hat_middle, poly_degree)
        poly_middle = np.poly1d(coeffs_middle_poly)
        y_hat_middle = poly_middle(x_vals)
        
        # Only keep pixels in image
        bool_array_valid_points = y_hat_middle <= n_rows
        x_vals_within_image = x_vals[bool_array_valid_points]
        y_vals_within_image = y_hat_middle[bool_array_valid_points]
    
        length = 0
        for pos, (x,y) in enumerate(zip(x_vals_within_image, y_vals_within_image)):
            pass
            # reached last value
            index_next_point = idx_curr_value =  pos+1
            if index_next_point == len(x_vals_within_image):
                break
            
            x_dist = np.absolute(x-x_vals_within_image[index_next_point])
            y_dist = np.absolute(y-y_vals_within_image[index_next_point])
            length +=np.sqrt(x_dist**2 + y_dist**2) # pythagoras     
            
    
        thickness = np.sum(mask)/length
        
        if debug:
            plt.title(f"method 1 center line\n layer length: {length:.4f} \n thickness: {thickness:.4f}")    
            plt.imshow(mask)
            plt.plot(x_vals_within_image, y_vals_within_image, '-',  linewidth=1)
            plt.show()
        
        return thickness, [coeffs_middle_poly]
    
    if method == 2: 
        #print("method 2 selected")
        # METHOD 2 - CALCULATE TOP AND BOTTOM EDGES AND AVERAGE DISTANCE
       
        list_layer_lengths = [] 
        
        if debug:
            plt.imshow(mask)
        
        # Average top and bottom lengths
        list_points_edges = []
        for equation_params in list_coeffs:
            pass
            coeffs, (min_value,max_value) = equation_params
            poly_eqn = np.poly1d(coeffs)
            
            # y values part of the image
            x_vals = np.linspace(0, n_cols, num=(n_cols +1)) # +1 to produce whole numbers/pixels
            y_hat = poly_eqn(x_vals)
            
            # Only keep pixels in image
            bool_array_valid_points = y_hat <= n_rows
            x_vals_within_image = x_vals[bool_array_valid_points]
            y_vals_within_image = y_hat[bool_array_valid_points]
            
            # store points in list_to_return
            list_points_edges.append((x_vals_within_image,y_vals_within_image))
            
            
            # Calculate top and bottom lengths
            length = 0
            for pos, (x, y) in enumerate(zip(*[x_vals_within_image, y_vals_within_image])) :
                pass
                index_next_point = idx_curr_value =  pos+1
    
                #last value outside of array, no next point
                if idx_curr_value == len(x_vals_within_image):
                    break
                # calculate distance
                x_dist = np.absolute(x-x_vals_within_image[index_next_point])
                y_dist = np.absolute(y-y_vals_within_image[index_next_point])
                length +=np.sqrt(x_dist**2 + y_dist**2) # pythagoras        
            list_layer_lengths.append(length)
 
            #plot image
            if debug:
                plt.plot(x_vals_within_image, y_vals_within_image, '-', label='estimated curve', linewidth=1)
    
        #calculate mean and finally plot lines
        mean_length = np.mean(list_layer_lengths)
        area = np.sum(mask)
        thickness = area/mean_length
        
        if debug:
            plt.title(f"method 2 avg top and bottom \n top layer: {list_layer_lengths[0]:.4f} bottom layer: {list_layer_lengths[1]:.4f} mean: {mean_length:.4f} \n thickness: {thickness:.4f} ")
            plt.show()
    
        # return thickness and coeffs of polynomials 
        return thickness, mean_length, area, [list_coeffs[0][0], list_coeffs[1][0]] , list_points_edges

#%%

if __name__ == "__main__":
    pass
    
# Algorithm --> Smooth curved surfaces
# for each mask -->
# to find the top and bottom
# 	for each column find top and bottom masks and take average

working_dir = Path("C:/Users/Nabiki/Desktop/split_masks_selected")
im_paths = list(working_dir.glob("*.tiff"))

#iterate through every mask
for im_path in im_paths: #[3:4]:
    pass
        # LOAD AND SHOW MASK
    mask = tifffile.imread(im_path)
    mask = mask.transpose() ################## image must be vertical and binary
    # plt.title("initial working image")
    # plt.imshow(mask)
    # plt.show()
    method = 2
    thickness = compute_layer_thickness(mask, method, show_images=False)
