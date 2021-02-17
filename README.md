# MachineLearning-Segmentation
Unet 2.5D model for image segmentation. Original resource: https://github.com/jocicmarko/ultrasound-nerve-segmentation

Two different 2.5D Slicing Techniques were teted independently:
1. 5 Slices in same plane:
in this technique, 5 slices from the center of each 3D image were obtained and used for training the model
2. 3 Orthogonal planes
in this technique, 3 slices from the center of each 3D image were obtained, however in 3 different axes, the x, y & z.A total of 9 slices were taken from each 3D image, 3 slices from each anatomical plane.
