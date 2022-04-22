import random

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import regularizers
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3,4'
random.seed(2020)
num_gpus = 2

##################
def basic_model(im_size = (128,128) ,num_filters = 64, k = 5, pool_size = 2, lr = 0.0001, num_gpus=4):
    cross_dev_ops = tf.distribute.ReductionToOneDevice(reduce_to_device="/device:CPU:0")
    if num_gpus == 8:
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_dev_ops,
                                              devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3","/gpu:4", "/gpu:5", "/gpu:6", "/gpu:7"])
    elif num_gpus == 4:
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_dev_ops,
                                                  devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:4"])
    elif num_gpus == 2:
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_dev_ops,
                                                  devices=["/gpu:0", "/gpu:1"])

    with strategy.scope():
        model = Sequential()
        model.add(
            Convolution2D(num_filters, k,
                          input_shape=(im_size[0], im_size[1], 3), padding='same'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        model.add(Convolution2D(num_filters * 2, k, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        model.add(Convolution2D(num_filters * 2, k, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        model.add(Convolution2D(num_filters * 4, k, padding='same'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        model.add(Convolution2D(num_filters * 4, k, padding='same'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        model.add(Convolution2D(num_filters * 8, k, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        model.add(Convolution2D(num_filters * 8, k, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        model.add(Flatten())
        model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.0001)))
        model.add(Dropout(0.5))
        model.add(Dense(256, kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.0001)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        opt = optimizers.Adam(lr=lr)

        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
