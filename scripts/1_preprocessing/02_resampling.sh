#!/bin/bash
#SBATCH --output=../logs/%j_resampling_.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

conda activate uia_seg

python 02_resampling.py "$@" 
