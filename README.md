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


## Setup

Use the commands in [setup.sh](setup.sh) to create a conda env and install the necessary libraries for the environemnt

## Preprocessing

### With nnUNet
The nnUnet repo has a command to preprocess the data, which will resample all the volumes to a common resolution and calculate normalization statistics for them (it also calculates other parameters for the experiments such as patch sizes and number of feature maps for the UNet model it will use).

The only necessary step is to have the data in the format specified in [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#dataset-folder-structure).

For example something like this, where the naming convention is Dataset<DATASET_ID>_DATASET_NAME :
```
nnUNet_raw/
├── Dataset001_USZBinaryAllVessels
│   ├── dataset.json   # JSON file describing classes of the dataset
│   ├── imagesTr
│   ├── imagesTs  # optional
│   ├── labelsTr  
│   └── labelsTs  # optional
├── Dataset002_USZBinaryAneurysmOnly
├── Dataset003_USZ3ClassesAneurysmVsHealthyVessels
├── Dataset004_USZ21Classes
├── Dataset005_ADAM3ClassesUntreatedUIAsVsTreatedUIAsVsBackground
├── Dataset006_ADAMBinaryUntreatedUIAsAndTreatedUIAsVsBackground
├── Dataset007_ADAMBinaryUntreatedUIAsVsBackground
├── ...
```

For each volume in `imagesTr`, we need the corresponding ground truth mask in `labelsTr`, which is why we have a dataset
for each type of groupings of the segmentation targets. 

To preprocess is then  a matter of running the command on the dataset you plan on using:
```bash
export nnUNet_raw="<Path to the dir with the dataset>"
export nnUNet_preprocessed="<Path to the dir where the preprocessed data will be saved>"
conda activate uia_seg

nnUNetv2_plan_and_preprocess --verify_dataset_integrity -d <DATASET_ID>  # 1 for USZ, 2 for ADAM for example
``` 

The results we have in this [document](documentation%2FSome%20information%20on%20the%20datasets%20and%20the%20task.pdf) are for these two datasets: `Dataset003_USZ3ClassesAneurysmVsHealthyVessels` and `Dataset007_ADAMBinaryUntreatedUIAsVsBackground`.

**Note**: For these results we did not apply a bias correction preprocessing step, which is something you might want to try out to improve the performance just with preprocessing.

### Without nnUNet

In case you do not want to use nnUNet, then you can preprocess by doing first the bias correction (time intensive) and then resampling the volumes

#### Bias correction
Use the script [01_bias_correction.py](scripts%2F1_preprocessing%2F01_bias_correction.py) as shown next:
```bash
conda activate uia_seg

python 01_bias_correction.py "$@" --multi_proc 
```
There is also a script to write them then to an h5 file.

#### Resampling
Use the script [02_resampling.py](scripts%2F1_preprocessing%2F02_resampling.py) as shown next:
```bash
conda activate uia_seg

python 01_bias_correction.py \

```

## Training

### With nnUNet
Similar to the plan and preprocess script, you have to use a command, specify the 

## Relevant repositories and papers

 1. [nnUnet repo](https://github.com/MIC-DKFZ/nnUNet) and [paper](https://www.nature.com/articles/s41592-020-01008-z)
 2. [UIA segmentation](https://github.com/kv13/UIASegmentation): Repository of a Semester Thesis from another student. He implemented a vanilla Unet and a Unet + Graph
autoencoder to segment the USZ dataset. This could be useful in case you do not want to use nnUnet.  
 3. [ADAM dataset challenge website](https://adam.isi.uu.nl/) and [paper](https://www.sciencedirect.com/science/article/pii/S1053811921004936) 
 4. [Implementation of the winner of the ADAM challenge](https://github.com/JunMa11/ADAM2020) and their [presentation](https://adam.isi.uu.nl/results/results-miccai-2020/participating-teams-miccai-2020/junma-2/)

