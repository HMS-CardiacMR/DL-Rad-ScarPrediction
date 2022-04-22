"""
This script combines radiomics and DL features to identify scar in non-Gd cine images
ALL features are already prepared off-line and ready to train the new model
"""
import os
import numpy as np
import random
import pandas as pd

import matplotlib.pyplot as plt
init_rs = 2021
random.seed(init_rs)

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

gpus = '0'
num_gpus = 1
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
# Disable warnings... jamming console
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
""" ###################  ALGORITHM PARAMETERS #####################"""
# choose testing datset you want: ext or internal
testing_dataset = 'EXTERNAL'  # 'EXTERNAL' or  'INTERNAL'
features_type   = 'combined'  # 'radiomics'   'combined'

if features_type == 'combined':
    req_num_rad_feats  = 'all'
else:
    req_num_rad_feats = 10

req_num_comb_feats = 20

sens_level = 0.90
cutoff_num_sl = 1  # minimum number of slices with scar required for a patient to be labeled POSITIVE

""" ###################  Deep Learning Features: Constants #####################"""
dn_dl = '../dl_features/'
fn_dl_train = {
      0: '/256_feats_training_XVAL_0.csv',
      1: '/256_feats_training_XVAL_1.csv',
      2: '/256_feats_training_XVAL_2.csv',
      3: '/256_feats_training_XVAL_3.csv',
      4: '/256_feats_training_XVAL_4.csv'    }
fn_dl_valid = {
      0: '/256_feats_validation_XVAL_0.csv',
      1: '/256_feats_validation_XVAL_1.csv',
      2: '/256_feats_validation_XVAL_2.csv',
      3: '/256_feats_validation_XVAL_3.csv',
      4: '/256_feats_validation_XVAL_4.csv'    }
fn_dl_int_test = {
      0: '/256_feats_testing_XVAL_0.csv',
      1: '/256_feats_testing_XVAL_1.csv',
      2: '/256_feats_testing_XVAL_2.csv',
      3: '/256_feats_testing_XVAL_3.csv',
      4: '/256_feats_testing_XVAL_4.csv'    }

fn_dl_ext_test = {
      0: '/256_feats_extTest_BIDMC100_XVAL_0.csv',
      1: '/256_feats_extTest_BIDMC100_XVAL_1.csv',
      2: '/256_feats_extTest_BIDMC100_XVAL_2.csv',
      3: '/256_feats_extTest_BIDMC100_XVAL_3.csv',
      4: '/256_feats_extTest_BIDMC100_XVAL_4.csv' }

if testing_dataset == 'EXTERNAL':
    fn_dl_test = fn_dl_ext_test
    testing_rad_fn = '../radiomics_data/all_features_ext_test_bidmc100.csv'
else: # 'INTERNAL'
    fn_dl_test = fn_dl_int_test  # choose testing datset you want
    testing_rad_fn = '../radiomics_data/all_features_internal_testing.csv'

# READ radiomics features: hold-out testing dataset
test_rad_table = pd.read_csv(testing_rad_fn, index_col=0)
pats_rad_ids = np.asarray(test_rad_table['pat_id'])

for cross_val in range(0,5):
    print('##############   Cross-validation ' + str(cross_val) + '   ########################')
    """############## RADIOMICS ####### Read dataset + Re-arrange patients to match DL data ##############"""
    #READ radiomics features from excel sheets for training and validation dataset
    training_rad_fn  = '../radiomics_data/all_features_train_XVAL_' + str(cross_val) + '.csv'
    validation_rad_fn= '../radiomics_data/all_features_valid_XVAL_'+ str(cross_val) + '.csv'
    train_rad_table = pd.read_csv(training_rad_fn, index_col=0)
    valid_rad_table = pd.read_csv(validation_rad_fn, index_col=0)
    # Re-arrange the patients: all positives first, then all negatives needed to match patients's sequence in deep learning dataset
    tpos_df = train_rad_table.loc[train_rad_table["slice_label"] == 1].copy()
    tneg_df = train_rad_table.loc[train_rad_table["slice_label"] == 0].copy()
    vpos_df = valid_rad_table.loc[valid_rad_table["slice_label"] == 1].copy()
    vneg_df = valid_rad_table.loc[valid_rad_table["slice_label"] == 0].copy()
    develop_rad_table = pd.concat([tpos_df,tneg_df,vpos_df,vneg_df]) # Now the table has patients in the same order as deep learning features
    develop_rad_table.reset_index(drop=True,inplace=True)
    """############# RADIOMICS ####### Select Best Features #####################"""
    # Determine the important features and remove all other features from input data
    y_dev_rad = develop_rad_table["slice_label"].copy()
    X_dev_rad = develop_rad_table.iloc[:,22:-2].copy() # These are columns representing pyradiomics diagnostics data generated: exclude

    for col in X_dev_rad.columns: # normalize features
        X_dev_rad[col] = (X_dev_rad[col]-develop_rad_table[col].min())/(develop_rad_table[col].max()-develop_rad_table[col].min()+0.000001)

    if req_num_rad_feats == 'all':
        X_test_rad = test_rad_table.iloc[:, 22:-2].copy()
        y_test_rad = test_rad_table['slice_label'].copy()
        pass # no need to do anything for development dataset
    else:
        lasso = LassoCV(random_state=init_rs, cv=5, verbose=False) # use LASSO for feature selection
        sfm = SelectFromModel(lasso, threshold=-np.inf, max_features = int(req_num_rad_feats)).fit(X_dev_rad, y_dev_rad)
        idx = [i[0] for i in np.argwhere(sfm.get_support() == True)]
        all_feats_names = X_dev_rad.columns
        selected_feats_names_rad = [all_feats_names[i] for i in idx]
        X_dev_rad = X_dev_rad[selected_feats_names_rad].copy()
        X_test_rad = test_rad_table[selected_feats_names_rad].copy()
        y_test_rad = test_rad_table['slice_label'].copy()

    for col in X_test_rad.columns:
        X_test_rad[col] = (X_test_rad[col] - develop_rad_table[col].min()) / (
            develop_rad_table[col].max() - develop_rad_table[col].min() + 0.000001)

    """############## DEEP-LEARNING ######### Read All Features  #####################"""
    dl_test_table  = pd.read_csv(dn_dl + fn_dl_test[cross_val], index_col=0)  # load DL features of the testing data generated by CNN developed using current cross_val split
    dl_train_table = pd.read_csv(dn_dl + fn_dl_train[cross_val], index_col=0) # load DL features of the training data generated by CNN developed using current cross_val split
    dl_valid_table = pd.read_csv(dn_dl + fn_dl_valid[cross_val], index_col=0) # load DL features of the validation data generated by CNN developed using current cross_val split
    dl_dev_table   = pd.concat([dl_train_table,dl_valid_table])
    dl_dev_table.reset_index(drop=True,inplace=True)
    y_dev_dl = dl_dev_table["slice_label"].copy()
    X_dev_dl = dl_dev_table.iloc[:,:-2].copy() # 2 columns are slice labels and patient IDs; exclude

    for col in X_dev_dl.columns: # normalize features
        X_dev_dl[col] = (X_dev_dl[col]-dl_dev_table[col].min())/(dl_dev_table[col].max()-dl_dev_table[col].min()+0.000001)
    y_test_dl = dl_test_table["slice_label"].copy()
    X_test_dl = dl_test_table.iloc[:,:-2].copy() # last 2 columns are slice labels and patient IDs; exclude

    for col in X_test_dl.columns: # normalize features
        X_test_dl[col] = (X_test_dl[col]-dl_dev_table[col].min())/(dl_dev_table[col].max()-dl_dev_table[col].min()+0.000001)

    """ ###### COMBINED ###### Combine radiomics-DL Features ##################"""
    aa_dev = X_dev_dl.copy()
    bb_dev = X_dev_rad.copy()
    aa_tst = X_test_dl.copy()
    bb_tst = X_test_rad.copy()

    test_pids_rad = np.asarray(test_rad_table['pat_id'])
    test_pids_dl = np.asarray(dl_test_table['pat_id'])
    test_pats_ids  = test_pids_dl          # PIDs Should be same Dl == RAD
    num_test_pats = len(set(test_pids_dl))

    if features_type == 'combined':
        X_dev = bb_dev.join(aa_dev)
        X_test = bb_tst.join(aa_tst)
    elif features_type == 'radiomics':
        X_dev = bb_dev
        X_test = bb_tst

    y_dev  = y_dev_rad  # y_dev_rad   both are same vector
    y_test = y_test_rad #y_test_rad   y_test_dl both are same vector

    """######################################################################"""
    """############# COMBINED ####### Select Best Features #####################"""
    # Determine the important features and remove all other features from input data
    if features_type == 'combined' or features_type == 'deep_learning':
        lasso = LassoCV(random_state=init_rs, cv=5, verbose=False)  # use LASSO for feature selection
        sfm = SelectFromModel(lasso, threshold=-np.inf, max_features=req_num_comb_feats).fit(X_dev,y_dev)
        idx = [i[0] for i in np.argwhere(sfm.get_support() == True)]
        all_feats_names = X_dev.columns
        selected_feats_names_rad = [all_feats_names[i] for i in idx]
        X_dev = X_dev[selected_feats_names_rad].copy()
        X_test = X_test[selected_feats_names_rad].copy()
        print("Most important features for combined model are:")
        print(*selected_feats_names_rad, sep="\n")
    """######################################################################"""

    ## MODEL
    clf = LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear', verbose= False)
    clf.fit(X_dev,y_dev)

    y_true, y_pred = y_test, clf.predict_proba(X_test)
    y_pred = y_pred[:, 1]

    print('#### Per-Slice Analysis: @Sesitivity =', sens_level)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    si = np.argwhere(tpr >= sens_level)[0]
    yy = y_pred >= thresholds[si]
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, yy).ravel()

    print('Slice Sens     : ', tp/(tp+fn))
    print('Slice Spec     : ', tn/(tn+fp))
    print('Slice Recall   : ', tn / (tn + fn + 0.00001))
    print('Slice Precision: ', tp / (tp + fp + 0.00001))
    print('Slice Accuracy : ', (tn+tp)/(tn+tp+fp+fn))
    auc = metrics.roc_auc_score(y_true,y_pred)
    print('Per-Slice Area-Under-Curve = ' + str(auc))

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Per Slice (area = {:.3f} )'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve - Slice LGE Predictions')
    plt.legend(loc='best')

    """ ######################################## Per-Patient Analysis ########################################"""
    # printing FN and TP for few thresholds
    Err = np.zeros([100, 1])
    pFP = np.zeros([100, 1])  # p for patient
    pFN = np.zeros([100, 1])
    pTP = np.zeros([100, 1])
    pTN = np.zeros([100, 1])
    pSN = np.zeros([100, 1])
    pSP = np.zeros([100, 1])
    pREC = np.zeros([100, 1])
    pPRE = np.zeros([100, 1])
    pACC = np.zeros([100, 1])

    cnt = 0
    done = 0
    for thresh in range(100):
        yy = y_pred >= thresh/100
        start = 0
        subj_pred = np.zeros([num_test_pats, 1])
        subj_real = np.zeros([num_test_pats, 1])
        subj_id = []
        for i in range(num_test_pats):
            num_slices = len(np.argwhere(test_pats_ids == test_pats_ids[start]))
            subj_res = yy[start:start + num_slices]  # predicted slice label (LGE yes/no)
            subj_label = y_true[start:start + num_slices]  # reference slice label (LGE yes/no)
            subj_id.append(test_pats_ids[start])

            if np.sum(subj_res) >= cutoff_num_sl:  # 2 slices at least have LGE+
                subj_pred[i] = 1
            if np.sum(subj_label) > 0:  # ground truth, 1 slice at least have LGE+
                subj_real[i] = [1]
            start = start + num_slices

        Err[cnt] = np.sum(np.abs(subj_pred - subj_real))
        pFN[cnt] = len(np.argwhere((subj_pred - subj_real) == -1))
        pFP[cnt] = len(np.argwhere((subj_pred - subj_real) == 1))
        pTN[cnt] = len(np.argwhere(subj_real + subj_pred == 0))
        pTP[cnt] = len(np.argwhere((subj_pred + subj_real) == 2))

        pSN[cnt] = pTP[cnt] / (pTP[cnt] + pFN[cnt])
        pSP[cnt] = pTN[cnt] / (pTN[cnt] + pFP[cnt])
        pREC[cnt] = pTN[cnt] / (pTN[cnt] + pFN[cnt]+0.00001)
        pPRE[cnt] = pTP[cnt] / (pTP[cnt] + pFP[cnt]+0.00001)
        pACC[cnt] = (pTP[cnt] + pTN[cnt]) / (pTP[cnt] + pFP[cnt] + pTN[cnt] + pFN[cnt])

        if (pSN[cnt]>=sens_level) and (done==0):
            print('#### Per-Patient Analysis: @Sensitivity= ', sens_level)
            print([x[0] for x in subj_pred])
            print('Patient Sens     : ', pSN[cnt])
            print('Patient Spec     : ', pSP[cnt])
            print('Patient Recall   : ', pREC[cnt])
            print('Patient Precision: ', pPRE[cnt])
            print('Patient Accuracy : ', pACC[cnt])
            done = 1

        cnt += 1

    AUC = metrics.auc(1 - pSP, pSN)
    print('Per-Patient AUC', AUC)
    plt.figure(2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(1 - pSP, pSN, label='Per-Patient (area = {:.3f} )'.format(AUC))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve - Patient LGE Prediction')
    plt.legend(loc='best')

plt.show()
