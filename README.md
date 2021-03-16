# Anatomy-guided Multimodal Registration by Learning Segmentation without Ground Truth: Application to Intraprocedural CBCT/MR Liver Segmentation and Registration

Bo Zhou, S. Kevin Zhou, Chi Liu, James S. Duncan

Medical Image Analysis (MedIA), 2021

[[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_DuDoRNet_Learning_a_Dual-Domain_Recurrent_Network_for_Fast_MRI_Reconstruction_CVPR_2020_paper.pdf)]

This repository contains the PyTorch implementation of APA2Seg-Net.

### Citation
If you use this code for your research or project, please cite:

    @article{zhou2021anatomy,
      title={Anatomy-guided Multimodal Registration by Learning Segmentation without Ground Truth: Application to Intraprocedural CBCT/MR Liver Segmentation and Registration},
      author={Zhou, Bo and Augenfeld, Zachary and Chapiro, Julius and Zhou, S. Kevin and Liu, Chi and Duncan, James S.},
      journal={Medical Image Analysis},
      volume={xxx},
      pages={xxx},
      year={2021},
      publisher={Elsevier}
    }


### Environment and Dependencies
Requirements:
* Python 3.7
* Pytorch 1.4.0
* scipy
* scikit-image
* pillow
* itertools

Our code has been tested with Python 3.7, Pytorch 1.4.0, CUDA 10.0 on Ubuntu 18.04.


### Dataset Setup
    .
    preprocess/MRI_SEG           # data setup for MRI segmentation (target domain) from CT (source domain)
    ├── train_MRI.txt
    │
    ├── train_CT.txt
    │
    ├── test_MRI.txt
    │
    ├── CT                        # contain CT training data (index by train_CT.txt)
    │   ├── IMG_CT_1.png     
    │   ├── IMG_CT_1_mask.png   
    │   ├── IMG_CT_2.png     
    │   ├── IMG_CT_2_mask.png 
    │   ├── ...
    │   ├── IMG_CT_N.png     
    │   └── IMG_CT_N_mask.png 
    │
    ├── MRI                       # contain both MRI training and testing data (index by train_MRI.txt and test_MRI.txt)
    │   ├── IMG_MRI_1.png     
    │   ├── IMG_MRI_1_mask.png   
    │   ├── IMG_MRI_2.png     
    │   ├── IMG_MRI_2_mask.png 
    │   ├── ...
    │   ├── IMG_MRI_M.png     
    │   └── IMG_MRI_M_mask.png          
    └── 


train_MRI.txt contains the .png file names with content of

    IMG_MRI_1.png 
    IMG_MRI_2.png
    IMG_MRI_3.png 
    ...
    IMG_MRI_K.png    


train_CT.txt contains the .png file names with content of

    IMG_MRI_1.png 
    IMG_MRI_2.png
    IMG_MRI_3.png 
    ...
    IMG_MRI_N.png  


train_CT.txt contains the .png file names with content of

    IMG_MRI_K+1.png 
    IMG_MRI_K+2.png
    IMG_MRI_K+3.png 
    ...
    IMG_MRI_M.png  


Each .mat should contain a W x W complex value matrix with kspace data in it, where W x W is the kspace size. 
Please note the variable name should be set as 'kspace_py'.
Then, please add the data directory './Data/' after --data_root in the code or scripts.

### To Run Our Code
- Train the model
```bash
python train.py --experiment_name 'train_DuDoRN_R4_pT1' --data_root './Data/' --dataset 'Cartesian' --netG 'DRDN' --n_recurrent 4 --use_prior --protocol_ref 'T1' --protocol_tag 'T2'
```
where \
`--experiment_name` provides the experiment name for the current run, and save all the corresponding results under the experiment_name's folder. \
`--data_root`  provides the data folder directory (with structure illustrated above). \
`--n_recurrent` defines number of recurrent blocks in the DuDoRNet. \
`--protocol_tag` defines target modality to be reconstruct, e.g. T2 or FLAIR. \
`--protocol_ref` defines modality to be used as prior, e.g. T1. \
`--use_prior` defines whether to use prior as indicated by protocol_ref. \
Other hyperparameters can be adjusted in the code as well.

- Test the model
```bash
python test.py --experiment_name 'test_DuDoRN_R4_pT1' --accelerations 5 --resume './outputs/train_DuDoRN_R4_pT1/checkpoints/model_259.pt' --data_root './Data/' --dataset 'Cartesian' --netG 'DRDN' --n_recurrent 4 --use_prior --protocol_ref 'T1' --protocol_tag 'T2'
```
where \
`--accelerations` defines the acceleration factor, e.g. 5 for 5 fold accelerations. \
`--resume` defines which checkpoint for testing and evaluation. \
The test will output an eval.mat containing model's input, reconstruction prediction, and ground-truth for evaluation.

Sample training/test scripts are provided under './scripts/' and can be directly executed.


### Contact 
If you have any question, please file an issue or contact the author:
```
Bo Zhou: bo.zhou@yale.edu
```