# Date: 2022-11-25
# Author: Lu Jiqiao, George
# Department : Polyu HTI
# ==============================================================================
'''This util provides static method for handling datasets and creat data pipeline'''

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

import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import imageio
from datetime import datetime
import nrrd
import matplotlib.pyplot as plt


def load_img(path:str, img_rows=256, img_cols=256) -> tuple(np.ndarray, np.ndarray):
    '''
    load the image sequence and downsample to fit the size required by u-net
    
    Args:
        path: absolute path of image sequence
        img_rows: height of resized picture
        img_cols: width of resized picture
    
    Return:
        tuple of img3D and imgs_for_train(resized sequence of imgs)
    '''
    img3D = np.empty(shape=(len(os.listdir(path)), 512, 512))
    imgs_for_train = np.empty(shape=(len(os.listdir(path)), img_rows, img_cols))
    k=0
    for s in os.listdir(path):
        imfile = os.path.join(path,s)
        dcminfo = pydicom.read_file(imfile)
        rawimg = dcminfo.pixel_array
        img = apply_modality_lut(rawimg,dcminfo)
        img3D[k,:,:] = img
        # downsample the image
        img = cv2.resize(img, (img_rows,img_cols), interpolation=cv2.INTER_AREA)
        imgs_for_train[k,:,:] = img
        k += 1
    imgs_for_train[imgs_for_train > 255] = 255
    imgs_for_train[imgs_for_train < 0] = 0
    return img3D, imgs_for_train

def load_mask(fileName,img_rows=256,img_cols=256) -> tuple(np.ndarray, np.ndarray):
    '''
    load the mask image sequence and downsample to fit the size required by u-net
    
    Args:
        path: absolute path of mask sequence
        img_rows: height of resized picture
        img_cols: width of resized picture
    
    Return:
        tuple of msk3D and mask_for_train(resized sequence of mask images)
    '''    
    readdata, header = nrrd.read(fileName)
    msk3D = np.einsum('jik->kij', readdata) # swap the axis to [batch, width, height]
    msk3D = msk3D[::-1,:,:]
    mask_for_train = np.empty(shape=(readdata.shape[2],img_rows,img_cols))
    for k in range(readdata.shape[2]):
        msk = msk3D[k,:,:]
        # downsample the mask image
        msk = cv2.resize(msk, (img_rows,img_cols), interpolation = cv2.INTER_AREA)
        msk[msk >= 0.5] = 1
        msk[msk < 0.5] = 0
        mask_for_train[k,:,:] = msk
    return msk3D, mask_for_train

def create_unet_ds():
    
def exampleloadcase(maskFolder, fileName, caseID, imgFolder, subFolder):
    # maskFolder = r'D:\HMAI\Data\Liver\Clinical\ManualSegMask'
    # fileName = r'Segmentation.seg.nrrd'
    # imgFolder = r'D:\HMAI\Data\Liver\Clinical\CT Liver\HCC only'
    # subFolder = r'ST0\1.250000\contrast\1'
    path = os.path.join(imgFolder, caseID, subFolder)
    print(path)
    # print(os.listdir(path))
    fName = os.path.join(maskFolder, caseID, fileName)
    msk3D, mask_for_train = load_mask(fileName)
    img3D, imgs_for_train = load_img(path)
    print(mask_for_train.shape, imgs_for_train.shape)
    fig, subplt = plt.subplots(1,2)
    subplt[0].imshow(np.rot90(imgs_for_train[120,:,:]),cmap='gray')
    subplt[1].imshow(np.rot90(mask_for_train[120,:,:]))
    plt.show()