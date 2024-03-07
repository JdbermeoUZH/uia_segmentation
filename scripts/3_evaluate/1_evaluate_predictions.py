import os
import sys
import glob
import argparse

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from skimage.metrics import hausdorff_distance 
from torchmetrics.functional.classification import dice, recall, precision

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..','uia_segmentation', 'src')))

from preprocessing.utils import binarize_segmentation

from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


#---------- default params
script_path = os.path.dirname(os.path.abspath(__file__))
dir_output_default = os.path.normpath(os.path.join(script_path, '../../../../data/results/'))
metrics = {
    'dice': lambda gt, pred: dice(pred, gt, ignore_index=0).item(),
    'mhd': lambda gt, pred: hausdorff_distance(
        np.array(gt), np.array(pred), method='modified'),
}
#----------


def preprocess_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Bias correction of scans')
    
    parser.add_argument('dir_gt', type=str)
    parser.add_argument('dir_pred', type=str)
    parser.add_argument('dir_output', type=str)
    parser.add_argument('aneurysm_label', type=int)
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = preprocess_cmd_args()
    
    os.makedirs(args.dir_output, exist_ok=True)
    
    # Get the filepaths of the gt volumes
    filepaths_gt = glob.glob(os.path.join(args.dir_gt, '*.nii.gz'))
    filepaths_gt.sort()
    filepaths_pred = glob.glob(os.path.join(args.dir_pred, '*.nii.gz'))
    filepaths_pred.sort()
    
    # Check the ids are the same
    assert len(filepaths_gt) == len(filepaths_pred)
    assert set([os.path.basename(fp).split('_')[0] for fp in filepaths_gt]) == \
        set([os.path.basename(fp).split('_')[0] for fp in filepaths_pred])
        
    # Loop over the volumes and calculate the metrics
    metrics_info = []
    for fp_gt, fp_pred in zip(filepaths_gt, filepaths_pred):
        # Load the gt and pred volumes
        vol_gt = nib.load(fp_gt)
        vol_pred = nib.load(fp_pred)
        
        # Calculate information about the volumes
        y_size, x_size, z_size = vol_gt.header.get_zooms()
        vol_gt = vol_gt.get_fdata()
        vol_pred = vol_pred.get_fdata()
    
        # Get field of view in each dimension
        y_fov = y_size * vol_gt.shape[0]
        x_fov = x_size * vol_gt.shape[1]
        z_fov = z_size * vol_gt.shape[2]
        
        # Total volume of the image
        total_volume = y_fov * x_fov * z_fov
        
        # Foreground volume
        fg_volume = np.sum(vol_gt > 0) * y_size * x_size * z_size
        
        # Aneurysm volume
        aneurysm_volume = np.sum(vol_gt == args.aneurysm_label) * y_size * x_size * z_size
        
        # Aneurysm approx diameter
        aneurysm_diameter = ((6 * aneurysm_volume) / np.pi)**(1/3)
        
        # Calculate the metrics
        vol_aneurysm_only_gt = binarize_segmentation(vol_gt, args.aneurysm_label)
        vol_aneurysm_only_pred = binarize_segmentation(vol_pred, args.aneurysm_label)
        
        vol_gt = torch.tensor(vol_gt).int()
        vol_pred = torch.tensor(vol_pred).int()
        metrics_values = {metric: metric_fn(vol_gt, vol_pred)
                          for metric, metric_fn in metrics.items()}
        
        # Calculate metrics only for the aneurysm label
        vol_aneurysm_only_gt = torch.tensor(vol_aneurysm_only_gt).int()
        vol_aneurysm_only_pred = torch.tensor(vol_aneurysm_only_pred).int()
        metrics_values_aneurysm = {
            f'{metric}_aneurysm_only': metric_fn(
                vol_aneurysm_only_gt, vol_aneurysm_only_pred) 
            for metric, metric_fn in metrics.items()
            }
                
        # Save the metrics
        metrics_info.append({
            'vol_id': os.path.basename(fp_gt).strip('.nii.gz'),
            'py': y_size,
            'px': x_size,
            'pz': z_size, 
            'ny': vol_gt.shape[0],
            'nx': vol_gt.shape[1],
            'nz': vol_gt.shape[2],
            'y_fov': y_fov,
            'x_fov': x_fov,
            'z_fov': z_fov,
            'total_volume': total_volume,
            'fg_volume': fg_volume,
            'aneurysm_volume': aneurysm_volume,
            'aneurysm_approx_diameter': aneurysm_diameter,
            **metrics_values,
            **metrics_values_aneurysm
        })    
    
    metrics_df = pd.DataFrame(metrics_info).set_index('vol_id').sort_index()
    metrics_df.to_csv(os.path.join(args.dir_output, 'metrics.csv'))
    
    # Print averages for the metrics
    print('Averages for the metrics over the test set\n'
          '--------------------------')
    metric_cols = list(metrics.keys())
    metric_cols += [f'{metric}_aneurysm_only' for metric in metrics.keys()]
    print(metrics_df[metric_cols].mean())