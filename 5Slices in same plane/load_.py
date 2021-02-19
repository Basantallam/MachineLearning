import os
import numpy as np
import nibabel

data_path = '//content//drive//MyDrive'
image_rows = int(182//2) 
image_cols = int(219//2) 



def load_train_data():
    imgs_train = np.load('/content/drive/MyDrive/numpy array files 2.5D/5 slices in same plane//imgs_train.npy')
    masks_train = np.load('/content/drive/MyDrive/numpy array files 2.5D/5 slices in same plane//masks_train.npy')

   
    return imgs_train, masks_train

def load_test_data():
    imgs_test = np.load('/content/drive/MyDrive/numpy array files 2.5D/5 slices in same plane//imgs_test.npy')
    masks_test = np.load('/content/drive/MyDrive/numpy array files 2.5D/5 slices in same plane//masks_test.npy')
    return imgs_test, masks_test

