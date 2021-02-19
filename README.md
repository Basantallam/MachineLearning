# MachineLearning-Segmentation
The Following model is a Unet 2.5D model for image segmentation which, we designed and used in Segmentation of Brain Lesions in MRI scans of Multiple Sclerosis Patients.
 Original resource: https://github.com/jocicmarko/ultrasound-nerve-segmentation

Two different 2.5D Slicing Techniques were teted independently:
1. 5 Slices in same plane:
in this technique, 5 slices from the center of each 3D image were obtained and used for training the model
2. 3 Orthogonal planes:
in this technique, 3 slices from the center of each 3D image were obtained, however in 3 different axes, the x, y & z.A total of 9 slices were taken from each 3D image, 3 slices from each anatomical plane.

After Comparing the mwtrics, especially the Dice Similarity Coefficeint, it appears that the 3 Orthogonal planes slicing approach shows better model performance compared to  5 Slices in same plane. There is definitely room for improvement of the model, but this is the best we have reached till now.
