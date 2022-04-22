import xlrd
import numpy as np
import scipy.io
from sklearn.metrics import roc_curve
from sklearn import metrics
from cnn_models import basic_model as scar_model

import time
import matplotlib.pyplot as plt
from random import seed
from random import shuffle

import random

import os
start = time.time()
random.seed(2020)

"""###############################################################################################################
                                                    CONSTANTS
#################################################################################################################"""
FULL_EXCEL_PATH = '../cineradiomics_features-dummy_fname.xls'
MATCHED_POSITIVE_EXCEL_PATH = '../LGE-Cine-dummy_fname.xls'
NEGATIVE_EXCEL_PATH = '../LGE_negative_subjects_all.xls'
cutoff_num_sl = 1
sens_level = 0.90

#reading the data
wb = xlrd.open_workbook(MATCHED_POSITIVE_EXCEL_PATH)
sheet  = wb.sheet_by_name('all_ID')
allids = sheet.col_values(1)[1:]
pos1_n = sheet.col_values(2)[1:]
pos2_n = sheet.col_values(3)[1:]
neg_ids = [x for x in allids if x not in pos1_n and x not in pos2_n]
IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128

# NETWORK PARAMETERS
NUM_FILTERS = 64
KERNEL_SIZE = 5
POOL_SIZE   = 2
STRIDE      = 2

"""###############################################################################################################
                                                    UTILITIES
#################################################################################################################"""
np.random.seed(2020)
lge_neg_list = np.random.randint(0,484) # shuffle/random patient IDs
lge_pos_list = np.random.randint(0,273)
tot_neg = 400 # total number of negative cases used for training
tot_pos = 200
test_list_neg = lge_neg_list[tot_neg:] # select first 400 for training, remaining for testing
test_list_pos = lge_pos_list[tot_pos:]

def excel_reader(excel_file, sheet_name):
    xsl_wb = xlrd.open_workbook(excel_file)
    xsl_sheet = xsl_wb.sheet_by_name(sheet_name)
    ids = xsl_sheet.col_values(0)[1:]
    names = xsl_sheet.col_values(1)[1:]
    return names, ids

positive = scipy.io.loadmat('../lge_positive_cine_data_dummy_fname.mat')
negative = scipy.io.loadmat('../lge_negative_cine_data_dummy_fname.mat')

negative_names, negative_ids = excel_reader(NEGATIVE_EXCEL_PATH, 'dummy_sheet')
negative_names = [x for cnt, x in enumerate(negative_names) if negative_ids[cnt] in neg_ids]
positive_names, positive_ids = excel_reader(MATCHED_POSITIVE_EXCEL_PATH, 'dummy_sheet')

cross_val = 4 # the desired model to be retrieved was pretrained using which cross validation split? 0,1,2,3,4
subset_used_4testing = 'testing'
fold = 5 # i.e. 5-fold cross-validation
# This is the same code as that used to split training and validation dataset
if subset_used_4testing== 'validation': # override the testing set with validation dataset
    test_list_pos = lge_pos_list[cross_val * int(tot_pos / fold):(cross_val + 1) * int(tot_pos / fold)]
    test_list_neg = lge_neg_list[cross_val * int(tot_neg / fold):(cross_val + 1) * int(tot_neg / fold)]
elif subset_used_4testing== 'training': # override and use training dataset
    test_list_pos = lge_pos_list[0:int(cross_val * (tot_pos / fold))] + lge_pos_list[
                                                                         (cross_val + 1) * int(tot_pos / fold):tot_pos]
    test_list_neg = lge_neg_list[0:int(cross_val * (tot_neg / fold))] + lge_neg_list[
                                                                         (cross_val + 1) * int(tot_neg / fold):tot_neg]
elif subset_used_4testing=='testing':
    pass # do not override the held-out testing dataset
##########################################
# test dataset
test_images_neg = []
test_labels_neg = []
test_images_pos = []
test_labels_pos = []
slice_per_patient_test = []
test_names_pos = []
test_names_neg = []
pat_id = []
for num in test_list_pos:
    if isinstance(positive_names[num], float):
        positive_names[num] = str(int(positive_names[num]))
    test_names_pos.append(positive_names[num])

    positive_data = np.array(positive[positive_names[num]])
    positive_labels = np.ones([np.shape(positive_data)[0]])
    # print(np.shape(positive_labels))
    slice_per_patient_test.append(np.shape(positive_data)[0])
    if num == test_list_pos[0]:# initial fill-in the list
        test_images_pos = positive_data
        test_labels_pos = positive_labels
    else:
        test_images_pos = np.concatenate((test_images_pos, positive_data), 0)
        test_labels_pos = np.concatenate((test_labels_pos, positive_labels), 0)
    for i in range(len(positive_labels)):
        pat_id.append(positive_names[num]) # repeat the same patient name for all slices

for num in test_list_neg:

    if isinstance(negative_names[num], float):
        negative_names[num] = str(int(negative_names[num]))
    test_names_neg.append(negative_names[num])

    negative_data = np.array(negative[negative_names[num]])
    negative_labels = np.zeros([6])
    slice_per_patient_test.append(np.shape(negative_data)[0])
    if num == test_list_neg[0]:
        test_images_neg = negative_data
        test_labels_neg = negative_labels
    else:
        test_images_neg = np.concatenate((test_images_neg, negative_data), 0)
        test_labels_neg = np.concatenate((test_labels_neg, negative_labels), 0)

    for i in range(len(negative_labels)):
        pat_id.append(negative_names[num]) # repeat the same patient name for all slices

test_images = np.concatenate((test_images_pos, test_images_neg), axis=0)
test_labels = np.concatenate((test_labels_pos, test_labels_neg), axis=0)
pat_id = np.asarray(pat_id)
test_names  = test_names_pos + test_names_neg

test_last_channel = np.multiply(test_images[:, :, :, 0], test_images[:, :, :, 1])
test_images = np.concatenate((test_images, np.expand_dims(test_last_channel, 3)), 3)

"""###############################################################################################################
                                                    Model
#################################################################################################################"""

optm_scar_model = scar_model(im_size=(128, 128), num_filters=64, k=5, pool_size=2, lr=0.0001)

"""###############################################################################################################
                                                    EVALUATION
#################################################################################################################"""
folder_name = './weight_dir/'
weight_filename = { 0: '/cross_val_0/validation-0-new-weights-improvement-88-0.79.hdf5',
                    1 : '/cross_val_1/validation-1-new-weights-improvement-98-0.77.hdf5',
                    2 : '/cross_val_2/validation-2-new-weights-improvement-69-0.75.hdf5',
                    3 : '/cross_val_3/validation-3-new-weights-improvement-84-0.80.hdf5',
                    4 : '/cross_val_4/validation-4-new-weights-improvement-90-0.70.hdf5' }

optm_scar_model.load_weights(folder_name + weight_filename[cross_val]) # model is also loaded automatically
results = optm_scar_model.predict(test_images) # you can save model here for use on single gpu device

''' Display Heat Maps: conv2d_3 is the best compromise between resolution and link to network output'''
import grad_cam as gc
import matplotlib.pyplot as plt
slices_ids = []
for i,n in enumerate(slice_per_patient_test):
    for ii in range(n):
        slices_ids.append(test_names[i])

for di in range(2,1088,5):
    tstimg = test_images[di,:,:,:]
    cam = gc.GradCAM(optm_scar_model, layerName='conv2d_3', classIdx=1) # constructor of class; use conv2d_3
    heatmap = cam.compute_heatmap(np.expand_dims(tstimg,axis=0))
    tstimg_0 = np.squeeze(tstimg[:,:,0])
    tstimg_norm = 255*(tstimg_0 - np.min(tstimg_0)) / (np.max(tstimg_0) - np.min(tstimg_0))
    (heatmap, output) = cam.overlay_heatmap(heatmap,tstimg_norm, alpha=0.65)
    plt.matshow(output)
    if test_labels[di]==1:
        plt.title('Outcome=Pos, PID={sid}, Prob={r}%'.format(sid=slices_ids[di], r=np.round(results[di]*100)))
    else:
        plt.title('Outcome=Neg, PID={sid}, Prob={r}%'.format(sid=slices_ids[di],r=np.round(results[di] * 100)))
    plt.matshow(tstimg_0, cmap='gray')
    plt.title('Outcome=Neg, PID={sid}, Prob={r}%'.format(sid=slices_ids[di],r=np.round(results[di] * 100)))

'''Display Activation'''
from keras import models
di = 66
tstimg = test_images[di,:,:,:]
layer_outputs = [layer.output for layer in optm_scar_model.layers[:12]]  # Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=optm_scar_model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activations = activation_model.predict(np.expand_dims(tstimg,axis=0)) # Returns a list of five Numpy arrays: one array per layer activation
layer1_activation = activations[1]
for i in range(0,0+2):
    plt.matshow(layer1_activation[0, :, :,i], cmap='gray') #viridis
plt.matshow(tstimg[:,:,0], cmap='gray')

## Display Walets and processed radiomics images:
import radiomics.imageoperations as imop
import SimpleITK as sitk
dd1 = sitk.GetImageFromArray(tstimg[:,:,0],isVector=True)
dd2 = sitk.GetImageFromArray(tstimg[:,:,1],isVector=True)
for decompositionImage, decompositionName, inputKwargs in imop.getWaveletImage(dd1, dd2):
    decompositionImage = sitk.GetArrayFromImage(decompositionImage)
    plt.matshow(decompositionImage, cmap='gray')

dd3 = sitk.GetImageFromArray(tstimg[:,:,0])
dd4 = sitk.GetImageFromArray(tstimg[:,:,1])
gen = imop.getGradientImage(dd3,dd4)
for gImage, gnm, aa in gen:
    gradImage = sitk.GetArrayFromImage(gImage)
    plt.matshow(gradImage, cmap='gray')

dd3 = sitk.GetImageFromArray(tstimg[:,:,0])
dd4 = sitk.GetImageFromArray(tstimg[:,:,1])
gen = imop.getLogarithmImage(dd3,dd4)
for gImage, gnm, aa in gen:
    lbpImage = sitk.GetArrayFromImage(gImage)
    plt.matshow(lbpImage, cmap='gray')

dd5=tstimg[:,:,0]
dd5 = np.repeat(dd5[:, :, np.newaxis], 5, axis=2)
dd5 = sitk.GetImageFromArray( dd5, isVector=False )
dd6=tstimg[:,:,1]
dd6 = np.repeat(dd6[:, :, np.newaxis], 5, axis=2)
dd6 = sitk.GetImageFromArray( dd6, isVector=False )
tt = {'sigma': [1, 2]}
gen1 = imop.getLoGImage(inputImage=dd5, inputMask=dd6, **tt)
for gImage, gnm, aa in gen1:
    logImage = sitk.GetArrayFromImage(gImage)
    plt.matshow(logImage[:,:,1], cmap='gray')


""" ####### Run the following code to Export the Deep Learning Features ##########
## These features will be used for the combined RADIOMICS and DL model

from keras import Model
import pandas as pd
optm_scar_model.load_weights(folder_name + weight_filename[cross_val]) # model is also loaded automatically
dname = '../2021-DetectScar_nonGd_ProjectData/dl_features/'
fname = '/256_feats_'+ subset_used_4testing + '_XVAL_' + str(cross_val) + '.csv'
new_model = Model(optm_scar_model.inputs, optm_scar_model.layers[-3].output)
new_model.summary() # ensure that the last two layers are omitted
new_model.set_weights(optm_scar_model.get_weights()[0:-2])
# To match radiomics data, exclude all images without manual segmetnaton (i.e. zero masks) 
idx = np.argwhere(np.all(test_images[:, :, :, 1] == 0, axis=(1, 2)))
if len(idx): # delete zeros slices
    test_images   = np.delete(test_images,idx,axis=0)
    test_labels = np.delete(test_labels, idx, axis=0)    
    pat_id = np.delete(pat_id, idx, axis=0)
results = new_model.predict(test_images)
col=['dl_feat_'+str(x) for x in range(results.shape[-1])]
col.append('slice_label')
col.append('pat_id')
res_dataframe = pd.DataFrame(np.concatenate([results, np.expand_dims(test_labels,1), np.expand_dims(pat_id,1)],axis=1), columns=col)
res_dataframe.to_csv(dname+fname)
###################################################################################################"""

print('Cross-Validation:  '+ str(cross_val))
y_true = test_labels
y_pred = results

print('#### Per-Slice Analysis: @Sesitivity =', sens_level)
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
si = np.argwhere(tpr >= sens_level)[0]
yy = y_pred >= thresholds[si]
tn, fp, fn, tp = metrics.confusion_matrix(y_true, yy).ravel()
print('Slice Sens     : ', tp / (tp + fn))
print('Slice Spec     : ', tn / (tn + fp))
print('Slice Recall   : ', tn / (tn + fn + 0.00001))
print('Slice Precision: ', tp / (tp + fp + 0.00001))
print('Slice Accuracy : ', (tn + tp) / (tn + tp + fp + fn))
auc = metrics.roc_auc_score(y_true, y_pred)
print('Per-Slice Area-Under-Curve = ' + str(auc))

fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Per Slice (area = {:.3f} )'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - Slice LGE Predictions')
plt.legend(loc='best')

result = np.zeros([np.shape(results)[0], 1])
Err = np.zeros([100, 1])
pFP = np.zeros([100, 1]) # p for patient
pFN = np.zeros([100, 1])
pTP = np.zeros([100, 1])
pTN = np.zeros([100, 1])

pSN = np.zeros([100, 1])
pSP = np.zeros([100, 1])
pREC = np.zeros([100, 1])
pPRE = np.zeros([100, 1])
pACC = np.zeros([100, 1])

cnt  = 0
done = 0
for thresh in range(100):

    for i in range(0, np.shape(results)[0]):
        # print(results[i,0])
        if results[i, 0] < (thresh / 100):
            result[i, 0] = 0
        else:
            result[i, 0] = 1

    start = 0
    subj_pred = np.zeros([len(slice_per_patient_test), 1])
    subj_real = np.zeros([len(slice_per_patient_test), 1])
    for i in range(len(slice_per_patient_test)):
        slices = slice_per_patient_test[i]
        subj_res   = result[start:start + slices]      # predicted slice label (LGE yes/no)
        subj_label = test_labels[start:start + slices] # reference slice label (LGE yes/no)
        # print(np.sum(subj_res))
        if np.sum(subj_res) >= cutoff_num_sl: # 1 slice at least have LGE+
            subj_pred[i] = 1
        if np.sum(subj_label) > 0: # ground truth, 1 slice at least have LGE+
            subj_real[i] = [1]
        start = start + slices

    Err[cnt] = np.sum(np.abs(subj_pred - subj_real))
    pFN[cnt] = len(np.argwhere((subj_pred - subj_real) == -1))
    pFP[cnt] = len(np.argwhere((subj_pred - subj_real) == 1))
    pTN[cnt] = len(np.argwhere(subj_real + subj_pred == 0))
    pTP[cnt] = len(np.argwhere((subj_pred + subj_real) == 2))

    pSN[cnt] = pTP[cnt]/(pTP[cnt]+pFN[cnt])
    pSP[cnt] = pTN[cnt]/(pTN[cnt]+pFP[cnt])
    pREC[cnt]= pTN[cnt]/(pTN[cnt]+pFN[cnt]+ 0.00001)
    pPRE[cnt]= pTP[cnt]/(pTP[cnt]+pFP[cnt]+ 0.00001)
    pACC[cnt]= (pTP[cnt]+pTN[cnt])/(pTP[cnt]+pFP[cnt]+pTN[cnt]+pFN[cnt])

    if (pSN[cnt] >= sens_level) and (done == 0):
        print('#### Per-Patient Analysis: @Sensitivity= ', sens_level)
        print('Patient Sens     : ', pSN[cnt])
        print('Patient Spec     : ', pSP[cnt])
        print('Patient Recall   : ', pREC[cnt])
        print('Patient Precision: ', pPRE[cnt])
        print('Patient Accuracy : ', pACC[cnt])
        done = 1

    cnt += 1

AUC= metrics.auc(1-pSP,pSN)
print('AUC-new',AUC)
plt.figure(2)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(1-pSP, pSN, label='Per PAtient (area = {:.3f} )'.format(AUC))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - Patient LGE Prediction')
plt.legend(loc='best')
plt.show()
