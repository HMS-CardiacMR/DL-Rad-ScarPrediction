# DL-Rad-ScarPrediction
This code provides details of the methods used to implement a combined DeepLearning-Radiomics model for identifying patients without scar using DL and Radiomics analyses of non-Gd bSSFP cine sequences. The data flow module was simplified to make it easier to understand but we assume that each researcher will have to adapt this module to the new data format.

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

### Data Format
This code assumes that the data/images for all patients are stored in a matlab workspace (.mat file). The variables in this workspace mat file have names = the patients identifier (e.g. name or MR). Each of these variables represent a 4D matrix of size = (N x W x H x K), where N is number of cine slices, WxH is the mimage size (128x128), and K=2 represents the last dimension of the matrix which contains two images: cine grayscale image and myocardium binary mask.
The ground truth (patient label: LGE+ or LGE-) are stored in xls sheets containing names of LGE+ patioents and LGE- patients.




