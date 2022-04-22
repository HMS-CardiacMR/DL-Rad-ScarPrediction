"""
This script extracts radiomics features and saves them for further use to identify scar in non-Gd cine images
"""
import os
import numpy as np
import scipy.io
import random
import time
from data_utilities.radiomics_utilities import extract_all_radiomics_features
start = time.time()
random.seed(2020)

gpus = '0'
num_gpus = 1
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

mat_data = scipy.io.loadmat('../cine_data_dummy_fname.mat')
data_keys = mat_data.keys()
vol_data = []
lbl_data = []
pid_data = []

for i, k in enumerate(data_keys):
    if k[0] != 'p':
        continue

    dum_vol = np.array(mat_data[k])
    dum_slNum = np.shape(dum_vol)[0]
    if k[-1]=='n':
        dum_lbl = np.zeros([dum_slNum])
    else:
        dum_lbl = np.ones([dum_slNum])
    dum_pid = np.repeat(k, dum_slNum)

    if len(vol_data):  # initial fill-in the list
        vol_data = np.concatenate((vol_data, dum_vol), 0)
        lbl_data = np.concatenate((lbl_data, dum_lbl), 0)
        pid_data = np.concatenate((pid_data, dum_pid), 0)
    else:
        vol_data = dum_vol
        lbl_data = dum_lbl
        pid_data = dum_pid
"""###############################################################################################################
                           Radiomics: Compute and Save the features - TESTING DATASET
   #################################################################################################################"""
params = {}
params['binWidth'] = 1
dataset_name = 'ext_test_bidmc100'
storage_dfn = './all_features_' + dataset_name + '.xlsx'
extract_all_radiomics_features(all_img=vol_data[:, :, :, 0], all_mask=vol_data[:, :, :, 1], all_labels=lbl_data[:],all_pid=pid_data[:],
                               storage_dfn=storage_dfn, dataset_name=dataset_name,
                               voxelspacing=np.asarray([1., 1., 1.]), params=params, manualnormalize=False)
