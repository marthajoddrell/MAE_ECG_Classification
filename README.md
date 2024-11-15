This project trains an MAE on ECG images that have been generated from publicly available time series data to extract features for downstream classification. The MAE will provide the basis for a foundation model. The MAE framework comes from mmpretrain (https://mmpretrain.readthedocs.io/en/dev/papers/mae.html).

1. Create conda environment for MAE training and install mmpretrain:

```
# Create conda environment
conda create --name openmmlab python=3.8 -y

# Activate conda environment
source activate openmmlab

# Install pytorch
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# Follow the instructions at Prerequisites — MMPretrain 1.2.0 documentation, 
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
pip install -U openmim && mim install -e .

# Test installation
python demo/image_demo.py demo/demo.JPEG resnet18_8xb32_in1k --device cpu
python demo/image_demo.py demo/demo.JPEG resnet18_8xb32_in1k --device gpu
```

2. Prepare dataset and config files

To use a custom dataset, files need to be organised either in (https://mmpretrain.readthedocs.io/en/dev/user_guides/dataset_prepare.html):
- Subfolder format
- Text annotation file format

Then adjust associated configuration files (https://mmpretrain.readthedocs.io/en/dev/user_guides/config.html). There are 4 types:
- Models
- Datasets
- Schedules
- Runtime

From the mmpretrain GitHub repository, you can select a model (https://github.com/open-mmlab/mmpretrain). For this application, the self-supervised MAE (CVPR'2022) was chosen. The GitHub repository contains all the configurations files however these will have been downloaded when installing mmpretrain in Step 1 (https://github.com/open-mmlab/mmpretrain/tree/main/configs/mae).

In this repository, the folder 'my_configs/training' contains the configurations used for this project training of MAE. 

3. To train the MAE on dataset run in command line; **be mindful that wherever you execute this command, that is where the work_dirs will be saved e.g. output**:

```
python tools/train.py /my_configs/training/mae_vit-base-16_8xb512-amp-coslr-300e_in1k.py
```

Output will be a folder 'work_dirs' containing:
- epoch.pth files
- last checkpoint file
- folder containing
  - log
  - vis_data folder
- config .py file

4. To run downstream classification task, config files are contained in this repository in 'my_configs/testing'.


