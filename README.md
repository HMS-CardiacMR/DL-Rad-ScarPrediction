# DL-Rad-ScarPrediction
This code is meant to provide details of the methods used to implement a combined DeepLearning-Radiomics model for identifying patients without scar using DL and Radiomics analyses of non-Gd bSSFP cine sequences.

## Overview of code structure
There are three sets of scripts; each serves a different model: 
### (1) Deep Learning scripts
PUBLIC_dl_train.py: main code to train the deep learning model
PUBLIC_cn_models.py: implementation of the CNN models/classifiers
PUBLIC_test_new.py: main code to test the pretrained deep learning model
PUBLIC_grad_cam.py: the code we used to display activation maps (source: https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/)

### (2) Radiomics scripts
PUBLIC_compute_radiomics_ext.py: code to extract radiomics features

### (3) Combined Radiomics/DeepLearning scripts
PUBLIC_DL_rad_merge.py: code to train and test radiomics model or combined DL-radiomics model
