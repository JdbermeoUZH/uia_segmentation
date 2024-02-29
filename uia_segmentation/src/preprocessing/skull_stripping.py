"""
This step may be added, though it runs the risk of removing some part of the vessels in the volume

The results so far are without skull stripping or bias correction.
"""
import os

import numpy as np
import nibabel as nib
from nipype.interfaces import fsl

from preprocessing.utils import read_all_from_nii


def skull_stripping(img_path, name, sk_dir):
    # Set up the BET interface object
    bet = fsl.BET()
    # Set the input image file path
    bet.inputs.in_file = img_path
    # Set the output file path and name
    output_path         = os.path.join(sk_dir, name + '_mask.nii.gz')
    bet.inputs.out_file = output_path
    # Set the fractional intensity threshold (0.1 by default)
    bet.inputs.frac = 0.05
    # Set the vertical gradient in fractional intensity below
    # which the brain is not removed (0.0 by default)
    bet.inputs.vertical_gradient = 0.0
    # Set the radius of curvature (mm) for final surface tessellation
    bet.inputs.radius  = 75
    #bet.robust        = True
    # Run the BET interface object
    bet.run()
    return output_path


def skstrip_bet_interface(name, img_path, seg_path, mask_path, sk_dir, save_logs, path_logs, lock, multipreproc,
                          flag=False):
    seg_nii = nib.load(seg_path)
    new_seg_path = os.path.join(sk_dir, name + '_seg.nii.gz')
    nib.save(seg_nii, new_seg_path)
    del seg_nii

    if mask_path != '':
        new_mask_path = skull_stripping(mask_path, name, sk_dir)
    else:
        new_mask_path = skull_stripping(img_path, name, sk_dir)

    new_img_path = os.path.join(sk_dir, name + '_tof.nii.gz')
    if flag == False:
        # save the initial image without any change
        img_nii = nib.load(img_path)
        nib.save(img_nii, new_img_path)
        del img_nii
    else:
        img_data, img_affine, img_header = read_all_from_nii(img_path)
        msk_data, msk_affine, msk_header = read_all_from_nii(new_mask_path)
        try:
            assert msk_data.shape == img_data.shape
        except AssertionError as msg:
            print('ERROR: Inside skstrip_bet_interface mask and init image mismatch')
            raise AssertionError
        img_data_new = np.where(msk_data > 0, img_data, 0)
        nib.save(nib.Nifti1Image(img_data_new, img_affine, img_header),
                     new_img_path)

    return new_img_path, new_seg_path, new_mask_path


