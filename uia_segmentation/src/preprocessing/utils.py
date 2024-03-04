"""
Utility functions for preprocessing    

It includes functions to:
 - Load filepaths scans and segmentations masks of different datasets 
"""


import os
import glob
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import nibabel as nib


predefined_transformations = {
    'BinaryAllVessels': lambda x: binarize_segmentation(x),
    'BinaryAneurysmOnly': lambda x: binarize_segmentation(x, only_specific_vessel_class=4),
    '3ClassesAneurysmVsHealthyVessels': lambda x: collapse_all_classes_except_one(x, class_not_to_collapse=4, value_to_collapse_to=1),
    '21Classes': lambda x: x,
    'BinaryUntreatedUIAsAndTreatedUIAsVsBackground': lambda x: binarize_segmentation(x),
    'BinaryUntreatedUIAsVsBackground': lambda x: binarize_segmentation(x, only_specific_vessel_class=1),
    '3ClassesUntreatedUIAsVsTreatedUIAsVsBackground': lambda x: x,
}


def binarize_segmentation(img_np, only_specific_vessel_class: int = None):
    if not only_specific_vessel_class:
        img_np = np.where(img_np >= 1, 1, 0)
    else:
        img_np = np.where(img_np == only_specific_vessel_class, 1, 0)
    return img_np


def collapse_all_classes_except_one(img_np, class_not_to_collapse: int, value_to_collapse_to: int):
    # Collapse all clasess to the class 1 except the class_not_to_collapse
    img_np = np.where((img_np != class_not_to_collapse) & (img_np != 0), 
                      value_to_collapse_to, img_np)
    
    # Set the class_not_to_collapse class to 2
    img_np = np.where(img_np == class_not_to_collapse, 2, img_np)
    
    return img_np.astype(np.int8)


def transform_label_vol(filepath: list, transformation: callable):
    img = nib.load(filepath)
    img_np = img.get_fdata()
    img_np = transformation(img_np).astype(np.int8)
    img = nib.Nifti1Image(img_np, img.affine, img.header)
    nib.save(img, filepath)
    


def read_all_from_nii(path_image_nii):
    nii_img = nib.load(path_image_nii)
    return nii_img.get_fdata(), nii_img.affine.copy(), nii_img.header.copy()


def find_largest_bounding_box_3d(arr):
    # Find the indices of non-zero elements
    z, y, x = np.nonzero(arr)

    # If there are no non-zero elements, return None or an appropriate response
    if len(z) == 0 or len(y) == 0 or len(x) == 0:
        return None

    # Find the minimum and maximum indices along each dimension
    top_left_front = (np.min(z), np.min(y), np.min(x))
    bottom_right_back = (np.max(z), np.max(y), np.max(x))

    # Return the coordinates of the bounding box
    return top_left_front, bottom_right_back


def get_filepaths_raw(
    path_to_tof_dir: str,
    fp_pattern_tof: Union[str, list[str]],
    path_to_seg_dir: Optional[str],
    fp_pattern_seg: Union[Optional[str], list[str]],
    level_of_dir_with_id: int,
    every_scan_has_seg: bool = True,
    ) -> OrderedDict[str, dict[str, str]]:
    """
    Get the filepaths of the scans and the segmentation masks of one the datasets
    
    Parameters
    ----------
    path_to_tof_dir : str
        Path to the folder where the TOF scans are stored
    fp_pattern_tof : str
        Regex pattern to define what files are TOF scans within the folder
    path_to_segmentation_dir : str
        Path to the folder where the segmentation masks are stored
    fp_pattern_seg : str
        Regex pattern to define what files are segmentation masks within the folder
    level_of_dir_with_id : int
        Level of the directory where the scan id is stored
    every_tof_has_seg : bool
        Whether every TOF scan has a segmentation mask or not
        
    Returns
    -------
    OrderedDict[str, dict[str, str]]
        Dictionary with the filepaths of the scans and the segmentation masks (optional)
    
    """
    
    scans_dict = OrderedDict()
    
    fp_pattern_tof = os.path.join(*fp_pattern_tof) if isinstance(fp_pattern_tof, list) else fp_pattern_tof
            
    # get the filepaths of the scans
    tof_scans_fps = glob.glob(os.path.join(path_to_tof_dir, fp_pattern_tof))
    
    for scan_fp in tof_scans_fps:
        scan_name = scan_fp.split(os.path.sep)[level_of_dir_with_id]
        scans_dict[scan_name] = {'tof': scan_fp}
        
    if path_to_seg_dir is not None:
        fp_pattern_seg = os.path.join(*fp_pattern_seg) if isinstance(fp_pattern_seg, list) else fp_pattern_seg
        seg_masks_fps = glob.glob(os.path.join(path_to_seg_dir, fp_pattern_seg))
        
        if every_scan_has_seg:
            assert len(tof_scans_fps) == len(seg_masks_fps), \
            "The number of scans and segmentation masks is not the same"
            
        # add the segmentation masks to the dictionary
        for seg_mask_fp in seg_masks_fps:
            scan_name = seg_mask_fp.split(os.path.sep)[level_of_dir_with_id]
            
            assert scan_name in scans_dict.keys(), \
                f"The scan {scan_name} does not have a TOF scan"
            scans_dict[scan_name]['seg'] = seg_mask_fp
            
    return scans_dict


def get_filepaths_preprocessed(
    path_to_dir: str,
    fp_pattern_tof: Union[str, list[str]] = ['*', '*_tof.nii.gz'],
    fp_pattern_seg: Union[str, list[str]] = ['*', '*_seg.nii.gz'],
    every_scan_has_seg: bool = True,
) -> OrderedDict[str, dict[str, str]]:
    """
    Get the filepaths of the preprocessed scans and the segmentation masks of one the datasets
    
    They always have the same structure:
        - {path_to_dir}/{scan_id}/{scan_id}_tof.nii.gz
        - {path_to_dir}/{scan_id}/{scan_id}_seg.nii.gz
    
    Parameters
    ----------
    path_to_dir : str
        Path to the folder where the preprocessed scans are stored
    tof_fp_pattern : str
        Regex pattern to define what files are TOF scans within the folder
    seg_fp_pattern : str
        Regex pattern to define what files are segmentation masks within the folder
    every_tof_has_seg : bool

        
    Returns
    -------
    OrderedDict[str, dict[str, str]]
        Dictionary with the filepaths of the preprocessed scans and the segmentation masks
    
    """
    
    scans_dict = get_filepaths_raw(
        path_to_tof_dir=path_to_dir,
        fp_pattern_tof=fp_pattern_tof if fp_pattern_tof is not None else ['*', '*_tof.nii.gz'],
        path_to_seg_dir=path_to_dir,
        fp_pattern_seg=fp_pattern_seg if fp_pattern_seg is not None else ['*', '*_seg.nii.gz'],
        level_of_dir_with_id=-2,
        every_scan_has_seg=every_scan_has_seg
    )        
    return scans_dict


def get_USZ_filepaths_raw(path_to_source_directory: str, include_segmentation_masks: bool = True) -> OrderedDict[str, dict[str, str]]:
    """
    Get the filepaths of the scans and the segmentation masks of the USZ dataset
    
    Parameters
    ----------
    path_to_source_directory : str
        Path to the folder where the scans are stored
    include_segmentation_masks : bool
        Whether to include the segmentation masks or not
    
    Returns
    -------
    OrderedDict[str, dict[str, str]]
        Dictionary with the filepaths of the scans and the segmentation masks (optional)
    
    """

    return get_filepaths_raw(
        path_to_tof_dir=path_to_source_directory,
        fp_pattern_tof=['*', '*_tof.nii.gz'],
        path_to_seg_dir=path_to_source_directory if include_segmentation_masks else None,
        fp_pattern_seg=['*', '*_seg.nii.gz'],
        level_of_dir_with_id=-2,
        every_scan_has_seg=True
    )


def get_ADAM_filepaths_raw(path_to_source_directory: str, include_segmentation_masks: bool = True) -> OrderedDict[str, dict[str, str]]:
    """
    Get the filepaths of the scans and the segmentation masks of the ADAM dataset
    
    Parameters
    ----------
    path_to_source_directory : str
        Path to the folder where the scans are stored
    include_segmentation_masks : bool
        Whether to include the segmentation masks or not
    
    Returns
    -------
    OrderedDict[str, dict[str, str]]
        Dictionary with the filepaths of the scans and the segmentation masks (optional)
    
    """

    return get_filepaths_raw(
        path_to_tof_dir=path_to_source_directory,
        fp_pattern_tof=['*', '*_TOF.nii.gz'],
        path_to_seg_dir=path_to_source_directory if include_segmentation_masks else None,
        fp_pattern_seg=['*', '*_aneurysms.nii.gz'],
        level_of_dir_with_id=-2,
        every_scan_has_seg=True
    )


def get_Lausanne_filepaths_raw(path_to_tof_dir: str, path_to_segmentation_dir: Optional[str]) -> OrderedDict[str, dict[str, str]]:
    """
    Get the filepaths of the scans and the segmentation masks of the Laussane dataset
    
    Note scan 482 does not have a segmentation mask, as it does not have an aneurysm
    
    Parameters
    ----------
    path_to_tof_dir : str
        Path to the folder where the TOF scans are stored
    path_to_segmentation_dir : str
        Path to the folder where the segmentation masks are stored
        
    Returns
    -------
    OrderedDict[str, dict[str, str]]
        Dictionary with the filepaths of the scans and the segmentation masks (optional)
    
    """

    return get_filepaths_raw(
        path_to_tof_dir=path_to_tof_dir,
        fp_pattern_tof=['*', '*', '*', '*_angio.nii.gz'],
        path_to_seg_dir=path_to_segmentation_dir,
        fp_pattern_seg=['*', '*', '*', '*Lesion_1_mask.nii.gz'],
        level_of_dir_with_id=-4,
        every_scan_has_seg=False
    )


def get_filepaths(
    preprocessed: bool = False,
    path_to_dir: str = None,
    dataset: str = None,
    path_to_tof_dir: str = None,
    fp_pattern_tof: Union[str, list[str]] = None,
    path_to_seg_dir: str = None,
    fp_pattern_seg: Union[str, list[str]] = None,
    level_of_dir_with_id: int = None,
    every_scan_has_seg: bool = True,
) -> OrderedDict[str, dict[str, str]]:
    
    if preprocessed:
        scans_dict = get_filepaths_preprocessed(
            path_to_dir=path_to_dir,
            fp_pattern_tof=fp_pattern_tof,
            fp_pattern_seg=fp_pattern_seg,
            every_scan_has_seg=every_scan_has_seg
        )
        
    elif dataset == 'USZ':
        scans_dict = get_USZ_filepaths_raw(path_to_source_directory=path_to_tof_dir)
    
    elif dataset == 'ADAM':
        scans_dict = get_ADAM_filepaths_raw(path_to_source_directory=path_to_tof_dir)
        
    elif dataset == 'Lausanne':
        scans_dict = get_Lausanne_filepaths_raw(path_to_tof_dir=path_to_tof_dir, 
                                                path_to_segmentation_dir=path_to_seg_dir)
    
    else:
        scans_dict = get_filepaths_raw(
            path_to_tof_dir=path_to_tof_dir,
            fp_pattern_tof=fp_pattern_tof,
            path_to_seg_dir=path_to_seg_dir,
            fp_pattern_seg=fp_pattern_seg,
            level_of_dir_with_id=level_of_dir_with_id,
            every_scan_has_seg=every_scan_has_seg
            )
        
    return scans_dict


