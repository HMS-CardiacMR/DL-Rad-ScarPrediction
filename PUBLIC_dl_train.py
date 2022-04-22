# Deep LEARNING PREDICTION MODEL
# Code originally prepared and used by Dr. Arafati

import os
import xlrd
import numpy as np
import scipy.io
import random
from cnn_models import basic_model as scar_model
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import time
from keras.callbacks import ModelCheckpoint
from random import shuffle
from random import seed

start = time.time()
random.seed(2020)

gpus = '0,1,2,3'
num_gpus = 4
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
"""###############################################################################################################
                                                    CONSTANTS
#################################################################################################################"""
FULL_EXCEL_PATH = '../cineradiomics_features-dummy_fname.xls'
MATCHED_POSITIVE_EXCEL_PATH = '../LGE-Cine-dummy_fname.xls'
NEGATIVE_EXCEL_PATH = '../LGE_negative_subjects_all.xls'

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
# TRAINING PARAMETERS
BATCH_SIZE = 8
LEARNING_RATE    = 0.0001
VALIDATION_STEPS = 200
EPOCHS = 100

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

def excel_reader(excel_file,sheet_name):
    wb = xlrd.open_workbook(excel_file)
    sheet = wb.sheet_by_name(sheet_name)
    ids = sheet.col_values(0)[1:]
    names = sheet.col_values(1)[1:]
    return names,ids

positive = scipy.io.loadmat('../lge_positive_cine_data_dummy_fname.mat')
negative = scipy.io.loadmat('../lge_negative_cine_data_dummy_fname.mat')

negative_names, negative_ids = excel_reader(NEGATIVE_EXCEL_PATH, 'dummy_sheet')
negative_names = [x for cnt, x in enumerate(negative_names) if negative_ids[cnt] in neg_ids]
positive_names, positive_ids = excel_reader(MATCHED_POSITIVE_EXCEL_PATH, 'dummy_sheet')

# 5-fold cross-validation
fold = 5
train_n = [0 for i in range(int(tot_neg - tot_neg / fold))]
train_p = [1 for i in range(int(tot_pos - tot_pos / fold))]
train_all = train_n + train_p
shuffle(train_all, seed(2))

val_n = [0 for i in range(int(tot_neg / fold))]
val_p = [1 for i in range(int(tot_pos / fold))]
val_all = val_n + val_p
shuffle(val_all, seed(2))

for cross_val in range(0,5):
    """###############################################################################################################
                                                        DATA
    #################################################################################################################"""
    train_list_pos = lge_pos_list[0:int(cross_val * (tot_pos / fold))] + lge_pos_list[
                                                                         (cross_val + 1) * int(tot_pos / fold):tot_pos]

    validation_list_pos = lge_pos_list[cross_val * int(tot_pos / fold):(cross_val + 1) * int(tot_pos / fold)]

    train_list_neg = lge_neg_list[0:int(cross_val * (tot_neg / fold))] + lge_neg_list[
                                                                         (cross_val + 1) * int(tot_neg / fold):tot_neg]

    validation_list_neg = lge_neg_list[cross_val * int(tot_neg / fold):(cross_val + 1) * int(tot_neg / fold)]

    train_images = []
    train_labels = []
    pos_cnt = 0
    neg_cnt = 0
    for n_p in train_all:
        if n_p: # LGE positive =1
            num = train_list_pos[pos_cnt] # num = index of array (correponding to a patient)
            pos_cnt += 1
            if isinstance(positive_names[num], float):
                positive_names[num] = str(int(positive_names[num]))
            positive_data = np.array(positive[positive_names[num]])
            positive_labels = np.ones([positive_data.shape[0]])
            if len(train_images):
                train_images = np.concatenate((train_images, positive_data), 0)
                train_labels = np.concatenate((train_labels, positive_labels), 0)
            else:
                train_images = positive_data
                train_labels = positive_labels
        else: # LGE negative patient
            num = train_list_neg[neg_cnt]
            neg_cnt += 1
            if isinstance(negative_names[num], float):
                negative_names[num] = str(int(negative_names[num]))
            negative_data = np.array(negative[negative_names[num]])
            negative_labels = np.zeros([6])
            if len(train_images):
                train_images = np.concatenate((train_images, negative_data), 0)
                train_labels = np.concatenate((train_labels, negative_labels), 0)
            else:
                train_images = negative_data
                train_labels = negative_labels

    train_last_channel = np.multiply(train_images[:, :, :, 0], train_images[:, :, :, 1]) # multiply LV_mask x image, set as channel 3
    train_images = np.concatenate((train_images, np.expand_dims(train_last_channel, 3)), 3)

    validation_images = []
    validation_labels = []
    pos_cnt = 0
    neg_cnt = 0
    for n_p in val_all:
        if n_p:
            num = validation_list_pos[pos_cnt]
            pos_cnt += 1
            if isinstance(positive_names[num], float):
                positive_names[num] = str(int(positive_names[num]))
            positive_data = np.array(positive[positive_names[num]])
            positive_labels = np.ones([positive_data.shape[0]])
            if len(validation_images):
                validation_images = np.concatenate((validation_images, positive_data), 0)
                validation_labels = np.concatenate((validation_labels, positive_labels), 0)
            else:
                validation_images = positive_data
                validation_labels = positive_labels
        else:
            num = validation_list_neg[neg_cnt]
            neg_cnt += 1
            if isinstance(negative_names[num], float):
                negative_names[num] = str(int(negative_names[num]))
            negative_data = np.array(negative[negative_names[num]])
            negative_labels = np.zeros([6])
            if len(validation_images):
                validation_images = np.concatenate((validation_images, negative_data), 0)
                validation_labels = np.concatenate((validation_labels, negative_labels), 0)
            else:
                validation_images = negative_data
                validation_labels = negative_labels

    validation_last_channel = np.multiply(validation_images[:, :, :, 0], validation_images[:, :, :, 1])
    validation_images = np.concatenate((validation_images, np.expand_dims(validation_last_channel, 3)), 3)

    """###############################################################################################################
                                                        TRAINING
    #################################################################################################################"""
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=30
    )
    test_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow(
        train_images,
        train_labels,
        shuffle=True,
        batch_size=BATCH_SIZE)

    """###############################################################################################################
                                                        VISUALIZATION
    #################################################################################################################"""
    log_dir = './tf-log/validation-{}'.format(cross_val)
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    cbks = [tb_cb]
    chkpt_filepath = './weight_dir/cross_val_{}/validation-{}'.format(cross_val, cross_val) + \
               "-new-weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(chkpt_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, tb_cb]

    optm_scar_model = scar_model(im_size = (128,128) ,num_filters = 64, k = 5, pool_size = 2, lr = 0.0001)
    training_history = optm_scar_model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch= np.round(0.8*600/BATCH_SIZE),
        validation_data= (validation_images,validation_labels),
        callbacks=callbacks_list,
        validation_steps= np.round(0.2 * 600 / BATCH_SIZE),
    )

    print("Average validation loss (slices): ", np.average(training_history.history['val_loss']))
    print("Average validation accuracy (slices): ", np.average(training_history.history['val_accuracy']))

    target_dir = './models/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    optm_scar_model.save(target_dir+'model-validation-{}.h5'.format(cross_val))
    optm_scar_model.save_weights(target_dir+'weights-validation-{}.h5'.format(cross_val))
