# UIA Segmentation

This repository contains some code to preprocess the TOF-MRA volumes (bias correction, resampling, write into h5 file), 
a pytorch Dataset class to load volumes from the h5 file, and a Unet model definition, which you can use to train your 
own UNet model.

However, if you are using the nnUNet framework, these are unnecessary, as nnUNet includes common preprocesing steps (except for bias correction). 
The results we have so far are with the default nnUnet parameters and without bias correction.

With nnUnet you just need to use the command line commands to preprocess the data, train a model, and evaluate it.
(see [preprocess_nnUNet.sh](scripts%2F1_preprocessing%2Fpreprocess_nnUNet.sh), [train_nnUNet.sh](scripts%2F2_training%2Ftrain_nnUNet.sh), and [predict_nnUNet.sh](scripts%2F3_evaluate%2Fpredict_nnUNet.sh))
The only requirement is that you have the volumes in the file structure that they require (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md#dataset-format).  

To use other loss functions with nnUnet or modify it's training, one can extend one of the Trainers defined in the repo: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md.

Lastly, for a description of the datasets and current benchmarks on the segmentation task please see: [Some information on the datasets and the task.pdf](documentation%2FSome%20information%20on%20the%20datasets%20and%20the%20task.pdf)


## Setup

Use the commands in [setup.sh](setup.sh) to create a conda env and install the necessary libraries for the environemnt
```bash
chmod +x ./setup.sh
./setup.sh
```

Or directly with the conda yaml:
```bash
conda env create -f environment.yml
```

## Preprocessing

### With nnUNet
The nnUnet repo has a command to preprocess the data and choose the achitecture of the UNet it will use. For the preprocessing, it applies common steps such as resampling all the volumes to a common resolution and calculate normalization statistics for them.

The only necessary step is to have the data in the format specified in [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#dataset-folder-structure).

For example something like this, where the naming convention is Dataset<DATASET_ID>_<DATASET_NAME> :
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

To preprocess a dataset is then a matter of running the command on the dataset you plan on using:
```bash
export nnUNet_raw="<Path to the dir with the dataset>"
export nnUNet_preprocessed="<Path to the dir where the preprocessed data will be saved>"
conda activate uia_seg

nnUNetv2_plan_and_preprocess --verify_dataset_integrity \
 -gpu_memory_target 6 \ # GB, default is 8 but it exceeds memory with the GPUs we have acess to
 -overwrite_plans_name 6GB_GPU \ # Name for the configuration/plan file generated to use during training
 -d <DATASET_ID>  # 1 for USZBinaryAllVessels, 2 for USZBinaryAneurysmOnly and so on
``` 

The results we have in this [document](documentation%2FSome%20information%20on%20the%20datasets%20and%20the%20task.pdf) are for these two datasets: `Dataset003_USZ3ClassesAneurysmVsHealthyVessels` and `Dataset007_ADAMBinaryUntreatedUIAsVsBackground`  (where the only non-default parameter is: -gpu_memory_target 6) 

### Without nnUNet

In case you do not want to use nnUNet, then you can preprocess by doing first the bias correction (time intensive) and then resampling the volumes to a common resolution.

#### Bias correction
Use the script [01_bias_correction.py](scripts%2F1_preprocessing%2F01_bias_correction.py) as shown next:
```bash
conda activate uia_seg

python 01_bias_correction.py --dataset USZ --path_to_tof_dir ../data/raw/USZ \
 --path_to_save_processed_data ../data/preprocessed/01_bias_correction/USZ
```
#### Resampling
Use the script [02_resampling.py](scripts%2F1_preprocessing%2F02_resampling.py) as shown next:
```bash
conda activate uia_seg

python 02_resampling.py --preprocessed --path_to_dir ../data/preprocessed/01_bias_correction/USZ \
 --voxel_size 0.3 0.3 0.6 --order 3 \
 --path_to_save_processed_data ../data/preprocessed/02_resampled/USZ

```

#### Write to h5 file

Use the script [04_save_into_hd5_file.py](scripts%2F1_preprocessing%2F05_save_into_hd5_file.py) to write the two datasets to a single h5 file with the same format (I stopped working on this task before fully testing this, so it might not work)

```bash
conda activate uia_seg

python 04_save_into_hd5_file.py USZ ../data/preprocessed/02_resampled/USZ \
 <Path to a JSON file that maps the label number to its name> \
 --num_folds 5 \
 --train_val_split 0.25 \
 --diameter_threshold 3 # Threshold to split the dataset into two groups <=3, and >3. Only creates cv fold for > diameter_threshold
```

## Training

### With nnUNet
Similar to the plan and preprocess script, you have to use a command to specify the dataset id you want to use during training. As nnUNet allows you to use differetnt types of UNet architectures to train (2d, 3d_lowres, 3d_full, 3d_casacade), you also need to specify this, as well as the fold you want to use (it always does cv5 by default), and the name of the "plan" generated in the previous step

```bash
export nnUNet_raw="<Path to the dir with the dataset>"
export nnUNet_preprocessed="<Path to the dir where the preprocessed data will be saved>"
export nnUNet_results="<Path to where the checkpoints and history will be stored>"

conda activate uia_seg

#nnUNet_n_proc_DA=0 use this prefix to not do multithreading. This helps with a memory issue that sometimes happens
nnUNet_n_proc_DA=0 nnUNetv2_train <DATASET_NAME_OR_ID> \
 <UNET_CONFIGURATION: 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres> \
 <Fold to use: 0-4> \
 -p 6GB_GPU \
 --npz  # Save softmax predictions from final validation as npz files (in addition to predicted segmentations). Needed for finding the best ensemble.
```

The results reported on the document are for `3d_fullres`.

### Without UNet
I do not have a script to train a model, but there is a [UNet Module](uia_segmentation/src/models/UNet.py) and [Dataset class](uia_segmentation/src/dataset/dataset_h5.py) (still needs more work) in this repo you could use to train your onw model if you want to.

Another student obtained similar results to nnUNet in the USZ dataset with a vanilla UNet and a rather small architecture: [2, 4, 16, 32] feature maps per resolution level. However, keep in mind that you can also specify this in the config files from nnUNet.

## Evalaution

You can use the predict method to obtain the predicted segmentation labels on the test set. Keep in mind that the dice calculated on the holdout set during training with nnUNet is an optimistic estimate, as it is not calculated over the volumes but over patches in a resampled resolution rather than the original one. The cleanest estimate of performance would be to obtain the prediction on the original resolution and at a volume level by obtaining the predictions with this command and then using your own script to calculate the metrics.

```bash
export nnUNet_raw="<Path to the dir with the dataset>"
export nnUNet_preprocessed="<Path to the dir where the preprocessed data will be saved>"
export nnUNet_results="<Path to where the checkpoints and history will be stored>"

conda activate uia_seg 

nnUNet_n_proc_DA=0 nnUNetv2_predict \
 -i ../data/nnUNet_raw/Dataset003_USZ3ClassesAneurysmVsHealthyVessels/ImagesTs \
 -o ../results/predictions/Dataset003_USZ3ClassesAneurysmVsHealthyVessels/ImagesTs \
 -d 3 \ # This is to retrieve the model that corresponds a specific dataset 
 -chk checkpoint_latest.pth \
 -p 6GB_gpu \
 -f all \ # Specify the folds of the trained model that should be used for prediction. Default: (0, 1, 2, 3, 4)
 --verbose    
```

## Relevant repositories and papers

 1. [nnUnet repo](https://github.com/MIC-DKFZ/nnUNet) and [paper](https://www.nature.com/articles/s41592-020-01008-z)
 2. [UIA segmentation](https://github.com/kv13/UIASegmentation): Repository of a Semester Thesis from another student. He implemented a vanilla Unet and a Unet + Graph
autoencoder to segment the USZ dataset. This could be useful in case you do not want to use nnUnet.  
 3. [ADAM dataset challenge website](https://adam.isi.uu.nl/) and [paper](https://www.sciencedirect.com/science/article/pii/S1053811921004936) 
 4. [Implementation of the winner of the ADAM challenge](https://github.com/JunMa11/ADAM2020) and their [presentation](https://adam.isi.uu.nl/results/results-miccai-2020/participating-teams-miccai-2020/junma-2/)

