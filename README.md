# UIA Segmentation

This repository contains some code to preprocess the TOF-MRA volumes (bias correction, resampling, write into h5 file), 
a pytorch Dataset class to load volumes from the h5 file, and a Unet model definition, which you can use to train your 
own UNet model.

However, if you are using the nnUnet framework, these are unnecessary with the exception of the bias correction step,
which is not included in the nnUNet preprocessing pipeline.

With nnUnet you just need to use the command line commands to preprocess the data, train a model, and evaluate it.
(see [preprocess_nnUNet.sh](scripts%2F1_preprocessing%2Fpreprocess_nnUNet.sh), [train_nnUNet.sh](scripts%2F2_training%2Ftrain_nnUNet.sh), and [predict_nnUNet.sh](scripts%2F3_evaluate%2Fpredict_nnUNet.sh))
The only requirement is that you have the volumes in the file structure that they require (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md#dataset-format).  

To use other loss functions with nnUnet or modify it's training, one can extend one of the Trainers defined in the repo: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md.

Lastly, for a description of the datasets and current benchmarks on the segmentation task please see: [Some information on the datasets and the task.pdf](documentation%2FSome%20information%20on%20the%20datasets%20and%20the%20task.pdf)


## Relevant repositories and papers

 1. [nnUnet repo](https://github.com/MIC-DKFZ/nnUNet) and [paper](https://www.nature.com/articles/s41592-020-01008-z)
 2. [UIA segmentation](https://github.com/kv13/UIASegmentation): Repository of a Semester Thesis from another student. He implemented a vanilla Unet and a Unet + Graph
autoencoder to segment the USZ dataset. This could be useful in case you do not want to use nnUnet.  
 3. [ADAM dataset challenge website](https://adam.isi.uu.nl/) and [paper](https://www.sciencedirect.com/science/article/pii/S1053811921004936) 
 4. [Implementation of the winner of the ADAM challenge](https://github.com/JunMa11/ADAM2020) and their [presentation](https://adam.isi.uu.nl/results/results-miccai-2020/participating-teams-miccai-2020/junma-2/)


## Setup

Use the commands in [setup.sh](setup.sh)