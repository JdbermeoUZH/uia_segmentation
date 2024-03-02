import os
import sys
import json
import shutil
import argparse

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'uia_segmentation', 'src')))

from preprocessing.utils import get_filepaths, predefined_transformations, transform_label_vol


#---------- default params
script_path = os.path.dirname(os.path.abspath(__file__))
default_output_dir = os.path.normpath(os.path.join(script_path, '../../../../data/nnUNet_raw/'))
#----------

def preprocess_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Bias correction of scans')
    
    parser.add_argument('dataset', type=str, choices=['USZ', 'ADAM'])
    parser.add_argument('dataset_id', type=int) 
    parser.add_argument('label_mapping', type=str, choices=predefined_transformations.keys())
    
    parser.add_argument('--test_set_idxs_fp', type=str)
    parser.add_argument('--dataset_name_suffix', type=str, default='')
    
    parser.add_argument('--preprocessed', action='store_true', default=False)
    parser.add_argument('--path_to_dir', type=str)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)   

    # Parameters when the dataset is not preprocessed into a common file structure   
    parser.add_argument('--path_to_tof_dir', type=str)
    parser.add_argument('--fp_pattern_tof', type=str, nargs='+')
    parser.add_argument('--path_to_seg_dir', type=str)
    parser.add_argument('--fp_pattern_seg', type=str, nargs='+')
    parser.add_argument('--level_of_dir_with_id', type=int, default=-2)
    parser.add_argument('--not_every_scan_has_seg', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.preprocessed:
        if args.path_to_dir is None:
            parser.error('--path_to_dir is required when --preprocessed is specified')  
    
    else:
        if args.path_to_tof_dir is None:
            parser.error('--path_to_tof_dir is required when --preprocessed is not specified')
                    
        if args.path_to_seg_dir is not None:
            if args.fp_pattern_seg is None:
                parser.error('--fp_pattern_seg is required when --path_to_seg_dir is not None')
        
        if args.dataset is None:
            if args.fp_pattern_tof is None:
                parser.error('--fp_pattern_tof is required when --dataset is None')
            
            if args.path_to_seg_dir is not None and args.fp_pattern_seg is None:
                parser.error('--fp_pattern_seg is required when --path_to_seg_dir is not None and --dataset is None')
                    
        if args.dataset == 'Lausanne' and args.path_to_seg_dir is None:
            parser.error('--path_to_seg_dir is required when --dataset is Lausanne')        
        
    return args


if __name__ == '__main__':
    args = preprocess_cmd_args()
    
    # Get filepaths of the the dataset
    scans_dict = get_filepaths(
        preprocessed=args.preprocessed,
        path_to_dir=args.path_to_dir,
        dataset=args.dataset,
        path_to_tof_dir=args.path_to_tof_dir,
        fp_pattern_tof=args.fp_pattern_tof,
        path_to_seg_dir=args.path_to_seg_dir,
        fp_pattern_seg=args.fp_pattern_seg,
        level_of_dir_with_id=args.level_of_dir_with_id,
        every_scan_has_seg=not args.not_every_scan_has_seg
    )
    
    # Load indexes of the images in the test set
    test_set_idxs_fp = args.test_set_idxs_fp or \
        os.path.join(script_path, f'test_idxs_{args.dataset.lower()}.json')
    test_idxs = json.load(open(test_set_idxs_fp))['test_idxs']
    
    # Create the directory for the Dataset
    dataset_name = args.dataset + args.label_mapping + args.dataset_name_suffix
    nn_unet_raw_dir = os.path.join(args.output_dir, f'Dataset{str(args.dataset_id).zfill(3)}_{dataset_name}')
    os.makedirs(nn_unet_raw_dir, exist_ok=True)
    
    # Create the ImagesTr, ImageTs, LabelsTr, and LabelsTs directories
    sub_dirs = ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs']
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(nn_unet_raw_dir, sub_dir), exist_ok=True)
        
    # Copy the dataset definition .json file
    shutil.copy(
        os.path.join(script_path, 'dataset_info', f'{args.label_mapping}.json'),
        os.path.join(nn_unet_raw_dir, 'dataset.json')
    )
    
    # Copy the files in the nnUNet format
    for scan_id, scan_dict in scans_dict.items():
        tof_fp = scan_dict['tof']
        seg_fp = scan_dict['seg']
        
        scan_name = f'{scan_id}_0000.nii.gz'
        seg_name = f'{scan_id}.nii.gz'
        
        if scan_id in test_idxs:
            scan_dir = os.path.join(nn_unet_raw_dir, 'imagesTs')
            seg_dir = os.path.join(nn_unet_raw_dir, 'labelsTs')
        else:
            scan_dir = os.path.join(nn_unet_raw_dir, 'imagesTr')
            seg_dir = os.path.join(nn_unet_raw_dir, 'labelsTr')
        
        shutil.copy(tof_fp, os.path.join(scan_dir, scan_name))
        shutil.copy(seg_fp, os.path.join(seg_dir, seg_name))
        
        # Transform the label volume
        transform_label_vol(
            filepath=os.path.join(seg_dir, seg_name),
            transformation=predefined_transformations[args.label_mapping]
        )
        
    