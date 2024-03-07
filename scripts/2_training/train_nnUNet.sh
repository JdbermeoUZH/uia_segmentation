#!/bin/bash
#SBATCH --output=../../logs/%j.out
#SBATCH --gres=gpu:1  # titan_xp, geforce_gtx_titan_x, geforce_rtx_2080_ti
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=3
#SBATCH --constraint='titan_xp|geforce_gtx_titan_x'

export nnUNet_raw="/scratch_net/biwidl319/jbermeo/data/nnUNet_raw"
export nnUNet_preprocessed="/scratch_net/biwidl319/jbermeo/data/nnUNet_preproc_new"
export nnUNet_results="/scratch_net/biwidl319/jbermeo/data/nnUnet_results_new"

source /scratch_net/biwidl319/jbermeo/conda/conda/etc/profile.d/conda.sh
conda activate /scratch_net/biwidl319/jbermeo/GNN-Domain-Generalization-main/net_scratch/conda_envs/tta_uia_seg
#conda activate uia_seg

#nnUNet_n_proc_DA=0 use  this prefix let's you choose the number of workers for the dataloaders
CUDA_LAUNCH_BLOCKING=1 nnUNet_n_proc_DA=0 nnUNetv2_train "$@" #DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz  # 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres 