from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from skimage.segmentation import mark_boundaries
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage.exposure import rescale_intensity
from keras.callbacks import History
from skimage import io
import tensorflow as tf
from tensorflow import keras

import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import imageio
from datetime import datetime
import nrrd
import matplotlib.pyplot as plt

smooth = 1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet(inputs=Input((256, 256, 1))):
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(learning_rate=1e-3), loss=dice_coef_loss, metrics=[dice_coef])
    #model.compile(optimizer=SGD(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])
    #model.compile(optimizer=SGD(lr=0.01), loss=dice_coef_loss, metrics=[dice_coef])
    
    return model

class Unet(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(None, 256, 256, 1), name='conv1_1')
        self.conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2')
        self.pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')

        self.conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1')
        self.conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2')
        self.pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')

        self.conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1')
        self.conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2')
        self.pool3 = MaxPooling2D(pool_size=(2, 2), name='conv1_1')

        self.conv4_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1')
        self.conv4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2')
        self.pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')

        self.conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')
        self.conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')

        self.conv2DT6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='conv2DT6')
        self.conv6_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6_1')
        self.conv6_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6_2')
        
        self.conv2DT7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='conv2DT7')
        self.conv7_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv7_1')
        self.conv7_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv7_2')

        self.conv2DT8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='conv2DT8')
        self.conv8_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv8_1')
        self.conv8_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv8_2')

        self.conv2DT9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='conv2DT9')
        self.conv9_1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv9_1')
        self.conv9_2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv9_2')

        self.conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='conv10')

    def call(self, input):
        x1 = self.conv1_1(input)
        x1 = self.conv1_2(x1)
        p1 = self.pool1(x1)

        x2 = self.conv2_1(p1)
        x2 = self.conv2_2(x2)
        p2 = self.pool2(x2)

        x3 = self.conv3_1(p2)
        x3 = self.conv3_2(x3)
        p3 = self.pool3(x3)

        x4 = self.conv4_1(p3)
        x4 = self.conv4_2(x4)
        p4 = self.pool4(x4)

        x5 = self.conv5_1(p4)
        x5 = self.conv5_2(x5)

        x6 = concatenate([self.conv2DT6(x5), x4], axis=3)
        x6 = self.conv6_1(x6)
        x6 = self.conv6_2(x6)

        x7 = concatenate([self.conv2DT7(x6), x3], axis=3)
        x7 = self.conv7_1(x7)
        x7 = self.conv7_2(x7)

        x8 = concatenate([self.conv2DT8(x7), x2], axis=3)
        x8 = self.conv8_1(x8)
        x8 = self.conv8_2(x8)

        x9 = concatenate([self.conv2DT9(x8), x1], axis=3)
        x9 = self.conv9_1(x9)
        x9 = self.conv9_2(x9)

        output = self.conv10(x9)
        return output
