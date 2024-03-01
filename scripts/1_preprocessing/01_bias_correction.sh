#!/bin/bash
#SBATCH --output=../logs/%j_preprocessing_bias_correction_.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1

conda activate uia_seg

python 01_bias_correction.py "$@" --multi_proc 
