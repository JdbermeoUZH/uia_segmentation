#!/bin/bash
#SBATCH --output=logs/%j.out
#SBATCH --gres=gpu:1  # titan_xp, geforce_gtx_titan_x, geforce_rtx_2080_ti
#SBATCH --cpus-per-task=4
#SBATCH --constraint='titan_xp|geforce_gtx_titan_x'

export nnUNet_raw="/scratch_net/biwidl319/jbermeo/data/nnUNet_raw"
export nnUNet_preprocessed="/scratch_net/biwidl319/jbermeo/data/nnUNet_preprocessed"
export nnUNet_results="/scratch_net/biwidl319/jbermeo/data/nnUnet_results"

conda activate uia_seg # pytcu11_8_py_3_9 - nn_unet_from_PyPi

#nnUNet_n_proc_DA=0 use this prefix to not do multithreading
nnUNet_n_proc_DA=0 nnUNetv2_train -p 6GB_GPU "$@" #DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz  # 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres 