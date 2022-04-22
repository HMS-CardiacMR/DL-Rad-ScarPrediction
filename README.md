# DL-Rad-ScarPrediction
This code provides details of the methods used to implement a combined DeepLearning-Radiomics model for identifying patients without scar using DL and Radiomics analyses of non-Gd bSSFP cine sequences.

## Overview of code structure
There are three sets of scripts; each serves a different model. 

### (1) Deep Learning scripts
PUBLIC_dl_train.py: main code to train the deep learning model
PUBLIC_cn_models.py: implementation of the CNN models/classifiers
PUBLIC_test_new.py: main code to test the pretrained deep learning model
PUBLIC_grad_cam.py: the code we used to display activation maps (source: https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/)
Code includes a commented part, that need to be run in order to export and save the DL features (as .csv file). This csv file will be read bythe combined DL-Radiomics module to train and test the DL-Radiomics model. 

### (2) Radiomics scripts
PUBLIC_compute_radiomics_ext.py: code to extract and save radiomics features (as .xlsx file). This csv file will be read bythe combined DL-Radiomics module to train and test the DL-Radiomics model.
PUBLIC_radiomics_utilities.py: code to implement radiomics function needed by PUBLIC_compute_radiomics_ext.py

### (3) Combined Radiomics/DeepLearning scripts
PUBLIC_DL_rad_merge.py: code to train and test radiomics model or combined DL-radiomics model

## Hyperparameters and Data (not provided):
(1) cine_data_dummy_fname.mat: matlab workspace/data file containing a list of matrices; one for each patient. The patient matrix is of size num_slices x W x H x 2, where the last dimension contains two images: cine image and myocardium mask in this imgae.




