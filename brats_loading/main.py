import os
import numpy as np
import nibabel as nib

from data_path import data_path

example_filename = os.path.join(data_path, 'BraTS20_Training_001_flair.nii')


img = nib.load(example_filename)

print(img.shape)
print(img.get_data_dtype() == np.dtype(np.int16))
print(img.affine.shape)


print("numpy")
#numpy version
data = img.get_fdata()
print(data.shape)
print(type(data))
print("break")


hdr = img.header
print(hdr.get_xyzt_units())


raw = hdr.structarr
print(raw['xyzt_units'])


print("Creating a new image")

data = np.ones((32, 32, 15, 100), dtype=np.int16)
img = nib.Nifti1Image(data, np.eye(4))
print(img.get_data_dtype() == np.dtype(np.int16))

print(img.header.get_xyzt_units())

nib.save(img, os.path.join('build', 'test4d.nii.gz'))


print(img.header)

