import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from nibabel.viewers import OrthoSlicer3D

img = nib.load('../data/COVID-19/COVID-19-CT-Seg_20cases/COVID-19-CT-Seg_20cases/coronacases_001.nii.gz')
OrthoSlicer3D(img.dataobj).show()
# figure, axes = plt.subplots()
# for i in range(len(img)):
#     axes.imshow(img[i])
# plt.show()
width, height, queue = img.dataobj.shape
num = 1
for i in range(0, 20):
    img_arr = img.dataobj[:, :, i]
    plt.subplot(5, 4, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1
plt.show()
