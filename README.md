# fetal_membrane_kayvan



### Testing and training ReLayNet
Our segmentations were performed  with the publicly available fully convolutional neural network ReLayNet originally intended for segmentation of retinal diabetic macular edema in optical coherence tomography images. A fork of this Unet with some modifications is available publicly on the following repository
https://github.com/skalalab/relaynet_pytorch


* **training_duke_dataset_to_h5** This script takes in the h5 dataset used in the main ReLayNet research article and converts it into h5 files that can be used to train the ReLayNet. This file was created as the dataset was not published in a format that could easily be used to train and predict on the ReLayNet.

* **training_placenta_dataset_to_h5** this script was created based on the script above to take our images and masks and package them into h5 files that can be used to train the ReLayNet. 


<hr>

## Processing Scripts
###  Inference on images 
Once we have a trained model, the **processing_inference_on_images** script loads the trained model and predicts layer segmentation on OCT tiff stacks and computes the layer thickness by using the function `compute layer thickness` found in the module **layer_edge_fitting_code**. The **processing_inference_on_images** outputs a tiff stack of layer segmentation masks in uin8 along with a tiff stack of 1) original image 2) layer segmentation and 3) layer edge estimation for each frame in the tiff stack, finally a csv file including 1) frame number 2) calculated thickness 3)area and 4)length for each of the 4 layers amnion, spongy,chorion, decidua.   
 
* **processing_inference_on_images**
  * **processing_layer_edge_fitting_code** func:`compute layer thickness`


### Apex rise vs pressure correlation and interpolation

* **processing_frame vs pressure vs apex rise code**
  * **Apex_rise_detection** : tracks apex rising throughout the sample by selecting a region to track taking it's fft and comparing it against the next frames fft to match regions. output array shows displacement of the membrane
  * **Apex_rise_crop_data** : this script loads previous apex rise detection and allows a user to  manually exclude refocusing regions on the apex rise graph as well as on the pressure graph. the resulting pressure graph is then interpolated to determine the pressure of the sample at any given frame 
  
### Reporting scripts

These reporting scrips were created to track samples being processed. They mainly look for pressure files (__Pressure.txt_) and make sure there is a corresponding imgage file. it then looks for output files for each of the tasks: segmentation, apex_rise vs pressure calculation and features.csv file containing the combined datasets. 

* **analysis_report_amniochorion** : Displays a list of all the samples left to process. Processing includes segmentation through ReLayNet, apex rise detection and pressure correlation. Additionally lines in this 
   
* **analysis_report_amnion_chorion** : Displays a list of all the samples left to process. Similar to the file above but does not use ReLayNet for segmentation.


<hr>
### Workflow 

 1. Train the ReLayNet by generating the training h5 files **training_placenta_dataset_to_h5**
    1. Resulting output will be a output_name.model file that can be used to predict on this model
 2. **processing_inference_on_images** this script loads the model and predicts layer segmentation and thickness based on a tiff stack of oct layers
 3. **Apex_rise_detection** and **Apex_rise_crop_data** scripts will in the  need dirctory **processing_frame vs pressure vs apex rise code** need to be run in matlab in this order. The **Apex_rise_detection** script will track change in apex position which will then be used in the  **Apex_rise_crop_data** to correlate and interpolate apex rise and pressure measurements from the arduino.
1. **merge_csv** once the **inference_on_images** and the **Apex_rise_detection** and **Apex_rise_crop_data** scripts have generated their corresponding **.csv** files. This script will merge them all into a **filename_features.csv** that contains all relevant data for analysis

<hr>
### Analysis 

* **analysis_cross_validation.py** Loads the data for one of the 10 folds the dataset was divided into. It then computes the layer segmentation prediction over the testing images and outputs a graph with dice coefficient compared to the ground truth of those test images.

**analysis_stress_tension_strain code** directory
* Original scripts were written in matlab (**Stress_Strain_Processing** **Tension_Strain_Processing**) code was then ported into python to automate processing and plotting of the toe and loaded regions for each of the apex rise vs pressure samples the main script for this is **tension_strain_processing** which loads information from the **features.csv** files prevoiusly generated

**analysis_stress_tension_strain code** directory

* **tension_strain_processing** this is the main script for loading and extracting the toe and loaded regions from the **features.csv** files previously generated

### Other files

* **relaynet_utils** a collection of functions used throughout the scripts to simplify processing and analysis