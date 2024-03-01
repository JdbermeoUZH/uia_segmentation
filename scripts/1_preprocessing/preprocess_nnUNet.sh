#!/bin/bash
#SBATCH --output=logs/%j.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

export nnUNet_raw="/scratch_net/biwidl319/jbermeo/data/nnUNet_raw"
export nnUNet_preprocessed="/scratch_net/biwidl319/jbermeo/data/nnUNet_preprocessed"
export nnUNet_results="/scratch_net/biwidl319/jbermeo/data"

conda activate uia_seg
nnUNetv2_plan_and_preprocess --verify_dataset_integrity "$@" #-d DATASET_ID
