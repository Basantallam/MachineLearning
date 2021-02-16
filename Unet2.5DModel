import os
import numpy as np
import nibabel

data_path = '//content//drive//MyDrive'
# we will undersample our training 2D images later (for memory and speed)

# rows and cols /2 because they will be undersampled by 2 later
image_rows = int(182//2)  
image_cols = int(219//2) 


def create_train_data():
    print('-'*30)
    print('Creating training data...')
    print('-'*30)
    #listing names of all training data into an array of strings
    #1st array is for input and the 2nd for labels (masks)
    train_data_path = os.path.join(data_path, 'ISBI Data//train')
    training_images = os.listdir(train_data_path)

    train_mask_data_path = os.path.join(data_path, 'ISBI Data//train_masks')
    training_masks = os.listdir(train_mask_data_path)

    #sorting file names alphabetically so they are correctly ordered  
    list.sort(training_images)
    list.sort(training_masks)

    #since we are doing 2.5D we will only use specific 2D images from the 3D image
    #the chosen 2D images will be appended to the following two arrays:

    #training images
    imgs_train = [] 
    #training masks 
    masks_train = [] 

    #looping over names of 3D images and extracting 2D images from them
    for i in range(0,len(training_images)):
      
        # every patient has 4 mri modalities with only 1 mask 
        # so "i" is divided by 4 in training_masks[i//4] so the same mask 
        # is used for with its 4 corresponding mri scans
        # we load 3D training masks
        training_mask = nibabel.load(os.path.join(train_mask_data_path,training_masks[i//4]))
        #we load 3D training image
        training_image = nibabel.load(os.path.join(train_data_path, training_images[i])) 
        
        #3D image is converted to a 3D numpy array
        #k is the index you  take the 2D image at.
        #In this approach we took 4 consecutive 2D images from the center of the 3Dimage
        for k in range((training_mask.shape[2]//2)-2,(training_mask.shape[2]//2)+2):
            #axial cuts are made along the z axis with undersampling by 2
            mask_2d = np.array(training_mask.get_fdata()[::2, ::2, k]) 
            image_2d = np.array(training_image.get_fdata()[::2, ::2, k])
            print("k="+str(k))
           #array of chosen 2D images of all patients appended together 
           #(Array of 2D arrays)
            masks_train.append(mask_2d)
            imgs_train.append(image_2d)
   
    #creating an empty numpy array 
        
    imgs = np.ndarray((len(imgs_train), image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((len(masks_train), image_rows, image_cols), dtype=np.uint8)


    # converting from list to numpy array
    for index, img in enumerate(imgs_train):
        imgs[index, :, :] = img
    # converting from list to numpy array
    for index, img in enumerate(masks_train):
        imgs_mask[index, :, :] = img

    #saving numpy array file   
    np.save('//content//drive//MyDrive//ISBI Data//imgs_train.npy', imgs)
    np.save('//content//drive//MyDrive//ISBI Data//masks_train.npy', imgs_mask)

    print('Saving training data to .npy files done.')

def create_test_data():
    print('-'*30)
    print('Creating test data...')
    print('-'*30)
    test_data_path = '//content//drive//MyDrive//test'

    tt=os.path.join(test_data_path, 'test')
    test_mask_path = os.path.join(test_data_path, 'mask')

    imgs_test = os.listdir(tt)     
    masks_test = os.listdir(test_mask_path) 

    #sorting file names alphabetically so they are correctly ordered  
    list.sort(imgs_test)
    list.sort(masks_test)

     

    #looping over names of 3D images and extracting 2D images from them
    for i in range(len(imgs_test)):
       # every patient has 3 mri modalities with only 1 mask 
        # so "i" is divided by 3 in masks_test[i//3] so the same mask 
        # is used for with its 3 corresponding mri scans

        print(image_name," ",mask_name)
        # we load 3D training masks
        img = nibabel.load(os.path.join(tt, masks_test[i//3]))
        msk = nibabel.load(os.path.join(test_mask_path , imgs_test[i]))

    #since we are doing 2.5D we will only use specific 2D images from the 3D image
    #the chosen 2D images will be appended to the following two arrays:

        imgs_testnpy=[]
        masks_testnpy=[]
          
        # 3D image is converted to a 3D numpy array
        # k is the index you  take the 2D image at.
        # In this approach we took 4 consecutive 2D images from the center of the 3Dimage
        for k in range((img.shape[2]//2)-2,(img.shape[2]//2)+2):
            print("k=",k)
            #axial cuts are made along the z axis with undersampling by 2
            img_2d = np.array(img.get_fdata()[::2, ::2, k])
            msk_2d = np.array(msk.get_fdata()[::2, ::2, k])

            # array of chosen 2D images of all patients appended together 
            #(Array of 2D arrays)
            imgs_testnpy.append(img_2d)
            masks_testnpy.append(msk_2d)

    # creating an empty numpy array 
    imgst = np.ndarray((len(imgs_testnpy), image_rows, image_cols), dtype=np.uint8)
    imgs_maskt = np.ndarray((len(masks_testnpy), image_rows, image_cols), dtype=np.uint8)
    
    # converting from list to numpy array
    for index, img in enumerate(imgs_testnpy):
        imgst[index, :, :] = img
        print(index)
    
    # converting from list to numpy array
    for index, img in enumerate(masks_testnpy):
        imgs_maskt[index, :, :] = img

    #saving numpy array file   
    np.save('/content/drive/MyDrive/ISBI Data/imgs_test.npy', imgst)
    np.save('/content/drive/MyDrive/ISBI Data/masks_test.npy', imgs_maskt)

    print('Saving test data to .npy files done.')

if __name__ == '__main__':
    create_train_data()
    create_test_data()
