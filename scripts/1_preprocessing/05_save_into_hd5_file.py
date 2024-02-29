"""
Script to save one of the three Aneurysm datasets as a hdf5 file. The data is preprocessed as follows:
- The scans are separated into two groups: < 4mm and >= 4mm. UIAs < 4mm are usually not treated, but we want to include down to 2.5 mm (models can still learn from them)
- The scans are resampled to a target size and resolution
- The scans are cropped around the label map (we were doing this to make the volumes smaller to try to process them whole in the GPUs, but you might want to skip this as we cannot replicate this preprocessing in a test sample)
- The scans are stored in a hdf5 file
- The filepaths of the scans with UIAs of less than 4mm of diameter are stored in the hdf5 file

"""

import os
import sys
import json
import argparse

import h5py
import numpy as np
import nibabel as nib
import nibabel.processing as nibp
from sklearn.model_selection import KFold, train_test_split

sys.path.append(os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'uia_segmentation', 'src')))

from preprocessing.utils import get_filepaths, find_largest_bounding_box_3d

#---------- paths & hyperparameters
num_folds_default                   = 5
train_val_split_default             = 0.25
preprocessed_default                = True
level_of_dir_with_id_default        = -2
path_to_save_processed_data_default = '/scratch_net/biwidl319/jbermeo/data/preprocessed/UIA_segmentation'
diameter_threshold_default          = 2.5  # mm, separate the scans into two groups: < 4mm and >= 4mm. UIAs < 4mm are usually not treated, but we want to include down to 2.5 mm (models can still learn from them)
seed                                = 0
num_channels                        = 1
target_size_default                 = (128, 220, 256)
target_resolution_default           = (1.0, 0.6, 0.6) # mm
crop_around_label_map_default       = True  # Default should normally be false unless we register the images, as their center moves a lot
#----------

bool_cmdl_arg = lambda x: str(x).lower() == 'true'

def preprocess_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Save as hdf5 file the preprocessed data for the UIA segmentation task')
    
    parser.add_argument('dataset', type=str, choices=['USZ', 'ADAM', 'Lausanne', None])
    parser.add_argument('path_to_dir', type=str, help='Path to the directory with the scans')
    parser.add_argument('path_label_names_dict', type=str, help='Path to the dictionary that maps the label names to their corresponding class')
    parser.add_argument('--target_size', type=int, nargs='+', default=target_size_default, help='Target size of the scans')
    parser.add_argument('--target_resolution', type=float, nargs='+', default=target_resolution_default, help='Target resolution of the scans')
    parser.add_argument('--crop_around_label_map', type=bool_cmdl_arg, default=crop_around_label_map_default, help='Crop the scans around the label map')
    parser.add_argument('--num_folds', type=int, default=num_folds_default, help='Number of folds for cross-validation')
    parser.add_argument('--train_val_split', type=float, default=train_val_split_default,
                        help='Percentage of the train set to use as validation')
    parser.add_argument('--diameter_threshold', type=float, default=diameter_threshold_default,
                        help='Diameter threshold to separate the scans into two groups: < 4mm and >= 4mm. UIAs < 4mm are usually not treated')
    parser.add_argument('--path_to_save_processed_data', type=str, default=path_to_save_processed_data_default)   
    parser.add_argument('--num_channels', type=int, default=num_channels, help='Number of channels of the scans')
        
    args = parser.parse_args()
    
    return args


def separate_filepaths_based_on_aneurysm_diameter(
    scan_fps: dict,
    diameter: float = diameter_threshold_default
    ):
    less_than_4mm = []
    for scan_id, image_fps in scan_fps.items():
        if 'seg' not in image_fps:
            continue
        seg_mask = nib.load(image_fps['seg'])
        seg_mask_arr = seg_mask.get_fdata()
        voxel_size = np.prod(seg_mask.header.get_zooms())
        
        # Calculate the approximate diameter of class 4, assuming it is a sphere
        uia_vol = seg_mask_arr[seg_mask_arr == 4].sum() * voxel_size
        
        if uia_vol == 0:
            continue
        
        uia_diameter = np.power(6 * uia_vol / np.pi, 1/3)
        
        if uia_diameter <= diameter:
            less_than_4mm.append(scan_id)    
            
    scan_fps_less_than_4mm = {scan_id: scan_fps[scan_id] for scan_id in less_than_4mm}
    scan_fps_greater_than_4mm = {scan_id: scan_fps[scan_id] for scan_id in scan_fps if scan_id not in less_than_4mm}    
    
    return scan_fps_less_than_4mm, scan_fps_greater_than_4mm    


def _verify_expected_channels(scan_data: np.ndarray, num_channels: int = 1):
    assert len(scan_data.shape) in [3, 4], \
    f'Expected a single channel or multichannel 3D scan, but got {len(scan_data.shape)}D'
    
    if num_channels == 1 and len(scan_data.shape) == 3:
        scan_data = np.expand_dims(scan_data, axis=0)
        
    elif num_channels == 1 and len(scan_data.shape) == 4:
        if scan_data.shape[0] == 1:
            scan_data = scan_data[0]
        else:
            raise ValueError(f'Expected a single channel scan, but got {scan_data.shape[0]} channels')
        
    return scan_data


def add_scans_to_group(
    scan_fps: dict, 
    h5_fp: str,
    group_name: str,
    every_scan_has_seg: bool,
    num_channels: int = 1,
    max_buffer_scans: int = 5,
    target_size: tuple = target_size_default,   # in DHW format
    target_resolution: tuple = target_resolution_default,   # in DHW format
    crop_around_label_map: bool = crop_around_label_map_default,
    ):
    num_scans_written = 0
    h5f = h5py.File(h5_fp, 'a')
    H5Data = h5f.create_group(group_name)
    
    for scan_id, image_fps in scan_fps.items():
        num_scans_written += 1
        seg_data = None
        
        if num_scans_written % max_buffer_scans == 0:
            h5f.close()
            h5f = h5py.File(h5_fp, 'a')
            H5Data = h5f[group_name]
            
        H5Scan = H5Data.create_group(scan_id)
        
        # Load the tof scan
        scan = nib.load(image_fps['tof'])   # in WHD format
        
        # Store original voxel and image size 
        px, py, pz = scan.header.get_zooms()
        H5Scan.create_dataset('px_orig', data=px)
        H5Scan.create_dataset('px_orig', data=py)
        H5Scan.create_dataset('px_orig', data=pz)
        
        nx, ny, nz = scan.shape
        H5Scan.create_dataset('nx_orig', data=nx)
        H5Scan.create_dataset('nx_orig', data=ny)
        H5Scan.create_dataset('nx_orig', data=nz)
        
        # Store the new voxel size
        if target_resolution is not None:
            pz, py, px = target_resolution
            H5Scan.create_dataset('px', data=px)
            H5Scan.create_dataset('py', data=py)
            H5Scan.create_dataset('pz', data=pz)
        
        # Load the segmentation, if available
        if every_scan_has_seg:
            seg_data = nib.load(image_fps['seg']).get_fdata()
        
        # Crop the scan around the label map, if specified
        if crop_around_label_map:
            if seg_data is None:
                raise ValueError('Crop around label map was specified, but no segmentation was found')

            # Get the bounding box of the segmentation
            bbox_min_coord, bbox_max_coord = find_largest_bounding_box_3d(seg_data)
            
            # Store the bounding box coordinates
            H5Scan.create_dataset('bbox_min_coord', data=bbox_min_coord)
            H5Scan.create_dataset('bbox_max_coord', data=bbox_max_coord)
            
            # Crop the scan around the bounding box
            scan_data = scan.get_fdata()
            scan_data = scan_data[bbox_min_coord[0]:bbox_max_coord[0], 
                                  bbox_min_coord[1]:bbox_max_coord[1],
                                  bbox_min_coord[2]:bbox_max_coord[2]]
            seg_data = seg_data[bbox_min_coord[0]:bbox_max_coord[0], 
                                bbox_min_coord[1]:bbox_max_coord[1], 
                                bbox_min_coord[2]:bbox_max_coord[2]]
            
            
        # Resample and resize the scan and segmentation, if necessary
        #  remember: scans are in WHD format and target_size and target_resolution are in DHW format
        if target_size is not None or target_resolution is not None:
            target_size = (target_size[2], target_size[1], target_size[0]) \
                if target_size is not None else scan.shape       
            target_resolution = (target_resolution[2], target_resolution[1], target_resolution[0]) \
                if target_resolution is not None else scan.header.get_zooms()        
        
            scan = nibp.conform(scan, voxel_size=target_resolution, out_shape=target_size, order=3, orientation='LPS')
            
            if seg_data is not None:
                seg_data = nibp.conform(seg_data, voxel_size=target_resolution, out_shape=target_size, order=0, orientation='LPS')

        # Convert the scan (and segmentation if available) from WHD to DHW and store it
        scan_data = scan.get_fdata()
        scan_data = _verify_expected_channels(scan_data, num_channels=num_channels)
        scan_data = np.swapaxes(scan_data, 0, 2)
        H5Scan.create_dataset('tof', data=scan_data, dtype=np.float32)
        
        if seg_data is not None:
            seg_data = _verify_expected_channels(seg_data, num_channels=num_channels)
            seg_data = np.swapaxes(seg_data, 0, 2)
        else:
            seg_data = np.zeros(scan.shape)
            
        H5Scan.create_dataset('seg', data=seg_data, dtype=np.uint8)


if __name__ == '__main__':
    args = preprocess_cmd_args()
    
    every_scan_has_seg = False if args.dataset == 'Lausanne' else True
    
    # Get filepaths of all scans
    scan_fps = get_filepaths(
        path_to_dir=args.path_to_dir, 
        preprocessed=preprocessed_default, 
        every_scan_has_seg=False)
    
    # Filter out the scans that have UIAs of less than 4mm of diameter
    scan_fps_leq_4mm, scan_fps = separate_filepaths_based_on_aneurysm_diameter(
        scan_fps, diameter=args.diameter_threshold if args.dataset != 'USZ' else 0)
    
    # Create a hdf5 file to store the preprocessed data
    os.makedirs(args.path_to_save_processed_data, exist_ok=True)
    h5_fp = os.path.join(args.path_to_save_processed_data, f'{args.dataset}.h5')
    h5f = h5py.File(h5_fp, 'w')
    
    # Store general metadata of the dataset
    # :================================================================================================:
    scan_ids = list(scan_fps.keys())
    h5f.create_dataset('ids', data=scan_ids)
    
    label_names = json.load(open(args.path_label_names_dict, 'r'))
    h5f.create_dataset('label_names', data=json.dumps(label_names))
    
    # Create a folds group to store the indexes of each of the folds
    # :================================================================================================:
    H5Folds = h5f.create_group('folds')
    
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(scan_ids)):
        # Create a group for each fold in the hdf5 file
        H5Fold = H5Folds.create_group(f'fold_{fold}')
        
        # Add the train and test indexes to the fold group
        H5Fold.create_dataset('train_idx', data=train_idx)
        H5Fold.create_dataset('test_idx', data=test_idx)
        
        # Create a list with the train-dev/val-dev folds
        train_dev_idx, val_dev_idx = train_test_split(
            train_idx, test_size=args.train_val_split, 
            shuffle=True, random_state=seed)    

        # Add the train-dev and val-dev indexes to the fold group
        H5Fold.create_dataset('train_dev_idx', data=train_dev_idx)
        H5Fold.create_dataset('val_dev_idx', data=val_dev_idx)
    
    h5f.close()
        
    # Create a data group to store the preprocessed data (each image index is a group)
    # :================================================================================================:
    add_scans_to_group(scan_fps, h5_fp, 'data', every_scan_has_seg)

    # Store the filepaths of the scans with UIAs of less than 4mm of diameter
    if len(scan_fps_leq_4mm) >= 0:
        add_scans_to_group(scan_fps_leq_4mm, h5_fp, 'data_leq_4mm', every_scan_has_seg)