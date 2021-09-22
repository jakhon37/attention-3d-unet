import os

import matplotlib.pyplot as plt

import nibabel as nib

from data_path import data_path

example_filename = os.path.join(data_path, 'BraTS20_Training_001_flair.nii')
epi_img = nib.load(example_filename)
img = epi_img.get_fdata()
print(img.shape)

def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = img[26, :, :]
slice_1 = img[:, 30, :]
slice_2 = img[:, :, 16]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")