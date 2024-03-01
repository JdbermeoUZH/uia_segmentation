conda create --name uia_seg python=3.10 -y
conda activate uia_seg
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia  -y
conda install pandas  -y
conda install scikit-learn  -y
conda install matplotlib  -y
conda install seaborn  -y
conda install h5py -y
conda install pyg -c pyg  -y
conda install scikit-image -y
pip install nibabel
pip install torchio
pip install opencv-python
pip install pynrrd
pip install itk
pip install dicom2nifti
pip install nipype
pip install nnunetv2
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
conda install graphviz -y
pip install IPython
pip install tqdm
pip install SimpleITK