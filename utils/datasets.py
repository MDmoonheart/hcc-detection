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
import tensorflow as tf
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import imageio
from datetime import datetime
import nrrd
import matplotlib.pyplot as plt
import db_connector as connector
import json


def create_dataset(img_dir: str, msk_dir: str, model: str):
    '''
    merge all static methods together to create dataset, the function exposes to outside as main function.
    
    Args:
        img_dir: main directory of all the cases, in which you can see all the cases files folder of all labels.
        msk_dir: directory of mask of images.
    
    Return:
        train_set: set for training (80% of datasets)
        test_set: set for validation (10% of datasets)
        validation_set: set for validation (10% of datasets)
    '''
    X, Y = get_raw_data(img_dir, msk_dir)
    X, Y = void_filter(X, Y)
    # filtered X and Y
    X, Y = void_filter(X, Y)
    # factorize X, Y into dataset
    if model.lower() == 'unet':
        ds = unet_factorize(X, Y)
    else:
        pass
    training_set, test_set, val_set = preprocess(ds)
    return training_set, test_set, val_set
    
def __init_conn(config_path='./config.json'):
    with open(config_path) as f:
        config = json.load(f)
    #unpack the parameters
    db_config = config.get('database_info')
    user = db_config.get('user')
    host = db_config.get('host')
    port = db_config.get('port')
    password = db_config.get('password')
    database = db_config.get('database')
    conn = connector.DBconnector(user=user, host=host, port=port, password=password, database=database)
    return conn

def load_img(path:str, img_rows=256, img_cols=256):
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

def load_mask(fileName, img_rows=256, img_cols=256):
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

def exampleloadcase(maskFolder, fileName, caseID, imgFolder, subFolder) -> None:
    # maskFolder = r'D:\HMAI\Data\Liver\Clinical\ManualSegMask'
    # fileName = r'Segmentation.seg.nrrd'
    # caseID = r'Case001'
    # imgFolder = r'D:\HMAI\Data\Liver\Clinical\CT Liver\HCC only'
    # subFolder = r'ST0\1.250000\contrast\1'
    path = os.path.join(imgFolder, caseID, subFolder)
    print(path)
    # print(os.listdir(path))
    fName = os.path.join(maskFolder, caseID, fileName)
    msk3D, mask_for_train = load_mask(fName)
    img3D, imgs_for_train = load_img(path)
    print(mask_for_train.shape, imgs_for_train.shape)
    fig, subplt = plt.subplots(1,2)
    subplt[0].imshow(np.rot90(imgs_for_train[120,:,:]),cmap='gray')
    subplt[1].imshow(np.rot90(mask_for_train[120,:,:]))
    plt.show()

def get_raw_data(img_dir: str, msk_dir: str):
    '''
    this method create dataset from images directory and mask directory using api of the database, it returns training sets 
    validation sets and test set with proportion 8 : 1 : 1.
    
    Args:
        img_dir: main directory of all the cases, in which you can see all the cases files folder of all labels.
        msk_dir: directory of mask of images.
    
    Return:
        train_set: set for training (80% of datasets)
        validation_set: set for validation (10% of datasets)
        test_set: set for validation (10% of datasets)
    '''
    file_name = r'Segmentation.seg.nrrd'
    cases = os.listdir(msk_dir) # extract the case id
    cases.remove("Case020_problem") # remove the invalid case
    cases.remove("Case056")
    cases.remove("Case067")
    cases.remove("Case083")
    cases.remove("Case148")
    cases.remove("Caselist.txt") # remove the txt file

    # concate the sql query and fetch all the subpath for the cases
    sql_tuple = "("
    for i, case in enumerate(cases):
        sql_tuple += f"'{case}'"
        if i != len(cases) - 1:
            sql_tuple += ','
    sql_tuple += ')'

    conn = __init_conn()
    sql = '''
    select 
        concat(c1.relative_path, '/', c1.case_name, c2.sub_path) 
    from 
        cases c1 join case_info c2 on(c1.cid = c2.case_id) 
    where 
        c1.case_name in %s and c2.phase = 1 and c2.thickness = 1.25
    order by c1.case_name
    ''' % sql_tuple
    temp = conn.execute_sql_query_command(sql)
    path = list(map(lambda x : x[0], temp))
    x_path = [img_dir + p for p in path]
    y_path = [msk_dir + r'/' + case + r'/' + file_name for case in cases]

    # conn.disconnect() # close the connerctor
    # load the img and mask accodingly
    x = tuple(map(lambda x : load_img(x)[1], x_path))
    x = np.vstack(x)
    y = tuple(map(lambda x : load_mask(x)[1], y_path)) 
    y = np.vstack(y)
    return x, y

def void_filter(imgs: np.ndarray, masks: np.ndarray):
    '''
    This method filter the invalid mask and its' images (mask pixel < 0)
    
    Args:
        imgs: images dataset
        mask: corrsponding mask of images sets
    
    Return:
        imgs_filtered: filtered images sets
        masks_filtered: filtered masks sets
    '''
    y_msk = []
    for i in range(masks.shape[0]):
        pic = masks[i]
        if(np.sum(pic) > 0):
            y_msk.append(i)
    masks_filtered = masks[y_msk, :, :]
    imgs_filtered = imgs[y_msk, :, :]
    return imgs_filtered, masks_filtered

def preprocess(ds:tf.data.Dataset, training_size=0.8, test_size=0.1):
    '''
    preprocess and split dataset into training sets, test sets, validation sets as specified portion.

    Args:
        ds: tf.dataset with X and Y.

    Return:
            train_set: set for training 
            validation_set: set for validation 
            test_set: set for validation 
    '''
    DATASIZE = ds.cardinality().numpy()
    TRAINING_SIZE = int(training_size * DATASIZE)
    TEST_SIZE = int(test_size * DATASIZE)
    VALIDATION_SIZE = DATASIZE - TRAINING_SIZE - TEST_SIZE

    def parse_function(image, label):
        image = tf.image.convert_image_dtype(image, tf.float16)
        label = tf.image.convert_image_dtype(label, tf.float16)
        return image, label

    # implement operation shuffle, cast, batch, prefetch
    ds = ds.shuffle(buffer_size=DATASIZE)
    ds = ds.map(parse_function, num_parallel_calls=2)
    train_set = ds.take(TRAINING_SIZE)
    test_set = ds.skip(TRAINING_SIZE).take(TEST_SIZE)
    val_set = ds.skip(TRAINING_SIZE + TEST_SIZE)
    print(f"training size: {train_set.cardinality().numpy()}, test size: {test_set.cardinality().numpy()},\
    validation size: {val_set.cardinality().numpy()}")
    return train_set, test_set, val_set

def unet_factorize(X: np.ndarray, Y: np.ndarray) -> tf.data.Dataset:
    '''
    factorize the X, Y into tf.data.Dataset with requeired channel dimensions
    '''
    return tf.data.Dataset.from_tensor_slices((X[..., np.newaxis], Y[..., np.newaxis])).batch(2)

