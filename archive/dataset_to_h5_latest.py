
from scipy.io import loadmat
import matplotlib.pylab as plt
import math
import numpy as np
import copy
import os
import re
import h5py
"""
This script generates the training dataset for ReLayNet from the Duke dataset. 
change the directory names when use. If you want to reduce the number of classes
to be classified, modify the places maked by TODOs. you can also change the 
w1 and w2 values for weight generation.
"""

#print(layers)
def get_annotated_img_index( layers):
    """
    find the index of images which has been annotated with labels and return 
    the index, the range of pixel being labeled. 
    return:
        @valid_img_index: index of annotated images as 1d list 
        @valid_range: list of 2 elements
    """
    valid_img_index = []
    valid_range = 0
    # just for information
    for img_i in range(len(layers[0,0,:])):
        for x in range(len(layers[:,0,0])):
            for y in range(len(layers[0,:,0])):
                if not math.isnan(layers[x,y,img_i]): 
                    if not img_i in valid_img_index:
                        valid_img_index.append(img_i)
                        valid_range = y
                    continue
            if img_i in valid_img_index:
                continue
    return valid_img_index, [valid_range, valid_range+523]
                 

def label_a_col(label_for_col):
    """
    takes in a 1d list indicating the layer transition of a column and recreate the actual image label
    with size of the image
    the endpoints inclusive or exclusive?
    """
    # SIZE: (496,523)
    col = np.ones(496) * 9 # TODO: 10 classes
    #col = np.ones(496) * 8 # 9 classes combine the bottom layer with background
    #col = np.ones(496) * 4
    for i in range(len(label_for_col)): # -4 creates 5 classes
        index = 7-i # label row index 0-7
        #index = 3-i # 5c
        if not math.isnan(label_for_col[index]):
            col[:int(label_for_col[index])] = index + 1 # class number 1-8
    return col
    

def remove_nan_from_label(label):
    """
    remove nans in the layer transition matrix to avoid erros in label_a_col by
    copying the value in the previous col.
    pass in label for a single 2d image
    """
    for row in range(len(label)):
        for col in range(len(label[0])):
            if math.isnan(label[row][col]):
                label[row][col] = label[row][col-1]
    return label
    
    
def create_labels(labels):
    """
    create labels for a 2d image
    """
    labels = remove_nan_from_label(labels)
    labels = np.swapaxes(labels,0,1)
    result = map(label_a_col, labels)
    output =[]
    for x in result:
        output.append(x)
    return np.swapaxes(output,0,1)



def add_fluid_label(total_label, fluid, valid_range):
    """
    fluid are images (3d array) 
    """
    img_index, temp = get_annotated_img_index( fluid)
    print(img_index)
    total = copy.deepcopy(total_label)
    img_index_for_total_label = 0
    
    fluid = fluid[:,valid_range[0]:valid_range[1],:]
    for k in img_index:
        for i in range(len(fluid[:,0,k])): # row
            for j in range(len(fluid[0,:,k])): # col
                if fluid[i,j,k] != 0:
                    # TODO: change the class number 10-10classes, 9-9classes
                    total[i,j,img_index_for_total_label] = 10 # the label fluid reagion should be
        img_index_for_total_label += 1
    return total
           
     
def print_annotated_index(dictionary):
    """
    gives which image has been labeled by human expert
    """
    layers = dictionary['manualLayers1'][:,:,:]
    valid_img_index = get_annotated_img_index(layers)
    print('manualLayers1 index of line with value: ',valid_img_index)
    
    matrix1 = get_annotated_img_index(dictionary['manualLayers2'][:,:,:])
    print('manualLayers2 index of line with value: ', matrix1)
    
    matrix2 = get_annotated_img_index(dictionary['manualFluid1'][:,:,:])
    print('manualFluid1 index of line with value: ', matrix2)
    
    matrix3 = get_annotated_img_index(dictionary['manualFluid2'][:,:,:])
    print('manualFluid2 index of line with value: ', matrix3)
          


def get_labels_3d(manuallayers, valid_range):
    """
    input manuallayers 1 or 2 to get the labeled masks for all the 11
    manually segmented images
    """

    layers = manuallayers[:,valid_range[0]:valid_range[1],:]
    print(valid_range[0], valid_range[1])
    valid_data_index, valid_range = get_annotated_img_index(manuallayers)
    label_output_3d = []
    for index in valid_data_index:
        try:
            output = create_labels(layers[:,:,index])  
        except ValueError:
            print('index ', index)
        label_output_3d.append(output)
        #####
        #plt.imshow(output)
        #plt.show()
        #####
    return label_output_3d


def get_labeled_img(images,valid_line,valid_range):
    """
    images = dictionary['images']
    """
    valid_images = []
    for index in valid_line:
        valid_images.append(images[:,valid_range[0]:valid_range[1],index])
    return valid_images
    

def weight(labels, w1, w2):
    """
    w(x) = 1 + w1 * I(|gradient(l(x))|) + w2 * I(l(x) = L)
    L = retinal layers and fluid masses
    :param labels: a list of labels
    :param w1:
    :param w2:
    :return:
    """    
    weights = np.zeros((labels.shape)) #(496,523,11)
    labels = np.array(labels)
    for i in range(labels.shape[2]):
        label = labels[:,:,i]
        I_w2 = label
        label = np.gradient(label)
        grad = list(map(abs, label))
        abs_grad = grad[0]+grad[1]
        abs_grad[abs_grad>0] = 1
        I_w1 = abs_grad
        pos1 = I_w2 == 1 # label of background
        pos2 = I_w2 == 9 #TODO:  background 
        I_w2 = 1*(~np.array(pos1 + pos2))
        weight = 1+ w1*I_w1 + w2*I_w2
        weights[:,:,i] = weight
        ####
        #plt.imshow(weights[:,:,i])
        #plt.show()
        #####
    return weights


    

if __name__ == '__main__': 
    # TODO: change the directory
    file_dir = '/home/skalalab/Desktop/relaynet_pytorch-master/datasets'
    #file_dir = '/Users/maywu/Desktop/Skala_lab/placenta_segmentation/datasets'
    images = np.zeros((496,523,1,110))
    labels = np.ones((496,523,2,110))
    start_pos = 0
    end_pos = 0
    files = os.listdir(file_dir)
    r = re.compile('.*\.mat')
    list_mat = list(filter(r.match, files))

    for mat_filename in list_mat:
        dictionary = loadmat(file_dir + os.sep + mat_filename)
        # a label for all manually segmented images
        valid_line, valid_range = get_annotated_img_index(dictionary['manualLayers1'])      
        pre_output_labels = get_labels_3d(dictionary['manualLayers1'], valid_range)
        pre_output_labels = np.moveaxis(pre_output_labels, 0, -1)
       
        output_label_with_fluid = add_fluid_label(pre_output_labels, dictionary['manualFluid1'],valid_range)
        end_pos = start_pos + 11
        
        # creating 4d array of labels
        labels[:,:,0,start_pos:end_pos] = output_label_with_fluid
        img = np.moveaxis(np.array(get_labeled_img(dictionary['images'],valid_line,valid_range)), 0, -1)
        #creating 4d array of images
        images[:,:,0,start_pos:end_pos] = img
        
        w = weight(output_label_with_fluid,3,6) #TODO: change w1 and w2
        labels[:,:,1,start_pos:end_pos] = w
        start_pos = end_pos

        for i in range(6):
            plt.imshow(images[:,:,0,i])
            plt.show()
            plt.imshow(labels[:,:,0,i])
            plt.show()
            plt.imshow(labels[:,:,1,i])
            plt.show()
    save_dir = '/Users/maywu/Desktop/Skala_lab/placenta_segmentation/datasets'
    with h5py.File('/home/skalalab/Desktop/relaynet_pytorch-master/Data.h5', 'w') as f:
        f.create_dataset('oct_dataset', data = images)  
    with h5py.File('/home/skalalab/Desktop/relaynet_pytorch-master/label.h5', 'w') as f:
        f.create_dataset('oct_labels', data = labels)  
        

        #for i in range(6):
        #    plt.imshow(images[:,:,0,i])
         #   plt.show()
         #   plt.imshow(labels[:,:,0,i])
         #   plt.show()
         #   plt.imshow(labels[:,:,1,i])
         #   plt.show()
    #save_dir = '/Users/maywu/Desktop/Skala_lab/placenta_segmentation/datasets'
    Set = np.ones((1,len(labels[0,0,0,:])))
    Set[0,int(len(Set[0,:])*0.8):] = 3 
    with h5py.File('/home/skalalab/Desktop/relaynet_unmodified/relaynet_pytorch-master/set.h5', 'w') as f:
        f.create_dataset('Set', data = Set) 
    with h5py.File('/home/skalalab/Desktop/relaynet_unmodified/relaynet_pytorch-master/Data.h5', 'w') as f:
        f.create_dataset('oct_dataset', data = images)  
    with h5py.File('/home/skalalab/Desktop/relaynet_unmodified/relaynet_pytorch-master/label.h5', 'w') as f:
        f.create_dataset('oct_labels', data = labels)  
    

    
