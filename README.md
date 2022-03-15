# OCT Measurement of mechanical properties in human fetal membranes

## Dataset Processing 

### OCT Layer Segmentation

Our segmentations were performed  with the publicly available fully convolutional neural network **ReLayNet** originally intended for segmentation of retinal diabetic macular edema in optical coherence tomography images. A fork of this Unet with some modifications is available publicly on the following repository
https://github.com/skalalab/relaynet_pytorch

Processing files for this task

* **Generating Duke Retina dataset for ReLayNet**  `training_duke_dataset_to_h5.py` :  This script takes in the h5 dataset used in the main ReLayNet research article and converts it into h5 files that can be used to train the ReLayNet. This file was created as the dataset was not published in a format that could easily be used to train and predict on the ReLayNet.

* **Generating fetal membrane dataset for ReLayNet** `training_placenta_dataset_to_h5.py` : this script was created based on the script above to take our images and masks and package them into h5 files that can be used to train the ReLayNet. 


* **Gereating inferences on images with trained model** `processing_inference_on_images.py` : Once we have a trained model, the `processing_inference_on_images.py` script loads the trained model and predicts layer segmentation on the OCT tiff stacks and computes the layer thickness by using the function `compute layer thickness` found in the module `processing_layer_edge_fitting_code.py`. The `processing_inference_on_images.py` outputs a tiff stack of layer segmentation masks in uint8 along with a tiff stack of 
1) original image 
2) layer segmentation 
3) layer edge estimation for each frame in the tiff stack

The script also exports a csv file including the following data for eahc of the four layers (amnion, spongy,chorion, decidua).
  1) frame number 
  2) calculated thickness 
  3) area
  4) length

### Apex Rise vs Pressure Correlation

Processing of the tiff stacks was done to extract the rise displacement from frame to frame. A separate script was then done to interpolate pressure over the frames kept in order to estimate the pressure at any one frame.

* **Extracting apex rise displacement** `Apex_rise_detection.m` : tracks apex rising throughout the sample by selecting a region on the image to track, taking it's fft and comparing it against the next frames fft to match regions. This script output a csv file of y displacements of the membrane called _samplename_raw_apex_rise.csv_
* **Apex rise vs pressure interpolation** `Apex_rise_crop_data.m` : this script loads previous apex rise detection and allows a user to  manually exclude refocusing regions on the apex rise graph as well as on the pressure graph. The resulting pressure graph is then interpolated to determine the pressure of the sample at any given frame. This scripts exports a csv called _samplename_Pressure_Apex.csv_
  
### Reporting scripts

These reporting scrips were created to track samples being processed. They mainly look for pressure files (_Pressure.txt_) and make sure there is a corresponding image file. It then looks for output files for each of the tasks: layer segmentation, apex rise vs pressure calculation and outputs if a sample is missing any. 

* **Report for Amniochrion samples** `processing_report_amniochorion.py` : Displays a list of all the samples left to process. Processing includes segmentation through ReLayNet, apex rise detection and pressure correlation. Additionally lines in this 
   
* **Report for Amnion and Chorion samples** `processing_report_amnion_chorion.py` : Displays a list of all the amnior or chorion samples left to process. Similar to the file above but does not look for the ReLayNet segmentation exports.

### Tension Strain

* **Computing Tension and Strain for Toe and Loaded Regions** `analysis_stress_tension_strain_code/tension_strain_processing.py` : Loads the apex_rise vs pressure exports (samplename_Pressure_Apex.csv),  
  *  
<hr>

### Workflow 

 1. Train the ReLayNet by generating the training h5 files **training_placenta_dataset_to_h5**
    1. Resulting output will be a output_name.model file that can be used to predict on this model
 2. **processing_inference_on_images** this script loads the model and predicts layer segmentation and thickness based on a tiff stack of oct layers
 3. **Apex_rise_detection** and **Apex_rise_crop_data** scripts will in the  need dirctory **processing_frame vs pressure vs apex rise code** need to be run in matlab in this order. The **Apex_rise_detection** script will track change in apex position which will then be used in the  **Apex_rise_crop_data** to correlate and interpolate apex rise and pressure measurements from the arduino.
1. **processing_merge_csv** once the **inference_on_images** and the **Apex_rise_detection** and **Apex_rise_crop_data** scripts have generated their corresponding **.csv** files. This script will merge them all into a **filename_features.csv** that contains all relevant data for analysis

_<python_file ==> export summary script>_
```
* **ReLayNet** ==> (processing_inference_on_images.py ==> *_thickness.csv)
  * length
  * area 
  * thickness

* **pressure files** (analysis_peak_pressure_extraction.py ==> *_max_pressure.csv)
  * max pressure (repeat across rows)
  * layer
  * location
  * pregnancy

* **Apex-rise vs Pressure** (tension_strain_processing.py ==> *toe_loaded_tension_strain.csv)
  * complete apex rise curve w/o NaN's  
  * loading curves
  * max_apex
  * max_strain
  * max_tension
  * toe modulus
  * linear modulus
  * frame_by_frame_tension_strain
  * toe_thresholds(low/high)
  * loaded_thresholds(low/high)

* **matlab scripts**
  * apex rise without refocusing cuts (*_Apex_raw.csv)

* **avg layer thicknesses** (analysis_thickness_toe_loaded_plots.py ==> *toe_loaded_thicknesses.py) 
 (amnion, chorion, spongy, decidua)
  * toe regions
  * linear regions
```
<hr>
### Analysis 

* **analysis_cross_validation.py** Loads the data for one of the 10 folds the dataset was divided into. It then computes the layer segmentation prediction over the testing images and outputs a graph with dice coefficient compared to the ground truth of those test images.

**analysis_stress_tension_strain code** directory
* Original scripts were written in matlab (**Stress_Strain_Processing** **Tension_Strain_Processing**) code was then ported into python to automate processing and plotting of the toe and loaded regions for each of the apex rise vs pressure samples the main script for this is **tension_strain_processing** which loads information from the **features.csv** files prevoiusly generated

**analysis_stress_tension_strain code** directory

* **tension_strain_processing** this is the main script for loading and extracting the toe and loaded regions from the **features.csv** files previously generated

### Other files

* **relaynet_utils** a collection of functions used throughout the scripts to simplify processing and analysis
