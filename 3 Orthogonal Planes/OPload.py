
import numpy as np

def load_train_data():
    imgs_train = np.load('/content/drive/MyDrive/numpy array files 2.5D/3 orthogonal planes/128/imgs_train.npy')
    masks_train = np.load('/content/drive/MyDrive/numpy array files 2.5D/3 orthogonal planes/128//masks_train.npy')
    return imgs_train, masks_train

def load_test_data():

    imgs_test = np.load('/content/drive/MyDrive/numpy array files 2.5D/3 orthogonal planes//128//imgs_test.npy')
    
    masks_test = np.load('/content/drive/MyDrive/numpy array files 2.5D/3 orthogonal planes/128/masks_test.npy')	
    return imgs_test, masks_test
