conda create --name uia_seg python=3.10 -y
conda activate uia_seg
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install nnunetv2 
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
conda install graphviz -y
pip install IPython
conda install h5py -y
pip install torchio
pip install opencv-python
pip install pynrrd
pip install itk
pip install dicom2nifti
pip install nipype
pip install torchmetrics