# Anatomy-guided Multimodal Registration by Learning Segmentation without Ground Truth: Application to Intraprocedural CBCT/MR Liver Segmentation and Registration

Bo Zhou, Zachary Augenfeld, Julius Chapiro, S. Kevin Zhou, Chi Liu, James S. Duncan

Medical Image Analysis (MedIA), 2021

[[Paper](https://www.sciencedirect.com/science/article/pii/S1361841521000876)]

This repository contains the PyTorch implementation of APA2Seg-Net.

### Citation
If you use this code for your research or project, please cite:

    @article{zhou2021anatomy,
      title={Anatomy-guided Multimodal Registration by Learning Segmentation without Ground Truth: Application to Intraprocedural CBCT/MR Liver Segmentation and Registration},
      author={Zhou, Bo and Augenfeld, Zachary and Chapiro, Julius and Zhou, S Kevin and Liu, Chi and Duncan, James S},
      journal={Medical Image Analysis},
      pages={102041},
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
    preprocess/MRI_SEG/PROC/       # data setup for MRI segmentation (target domain) from CT (source domain)
    ├── train_MRI.txt
    │
    ├── train_CT.txt
    │
    ├── test_MRI.txt
    │
    ├── DCT                        # contain CT training data (index by train_CT.txt)
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

IMG_CT_N.png is a 2D image and IMG_CT_N_mask.png is its segmentation.

For training, please specify the training data directory in the code options using: \
`--raw_A_dir` provides the domain A image data folder directory. In MRI segmentation example, it should be ./preprocess/MRI_SEG/PROC/DCT/ . \
`--raw_A_seg_dir` provides the domain A image's segmentation data folder directory. It should be identical to above, which is ./preprocess/MRI_SEG/PROC/DCT/ . \
`--sub_list_A` provides the directory of .txt file containing domain A's image file names. In MRI segmentation example, it should be ./preprocess/MRI_SEG/PROC/train_DCT.txt . \
`--raw_B_dir` provides the domain B image data folder directory. In MRI segmentation example, it should be ./preprocess/MRI_SEG/PROC/MRI/ . \
`--raw_B_seg_dir` provides the domain B image's segmentation data folder directory. It should be identical to above, which is ./preprocess/MRI_SEG/PROC/MRI/ . \
`--sub_list_B` provides the directory of .txt file containing domain B's image file names. In MRI segmentation example, it should be ./preprocess/MRI_SEG/PROC/train_MRI.txt . 

For testing, please specify the test data directory in the code options using: \
`--test_B_dir` provides the domain B test image data folder directory. In MRI segmentation example, it should be ./preprocess/MRI_SEG/PROC/MRI/ . \
`--test_img_list_file` provides the directory of .txt file containing domain B's test image file names. In MRI segmentation example, it should be ./preprocess/MRI_SEG/PROC/test_MRI.txt . \
`--test_seg_ouput_dir` provides the prediction output directory. 


### To Run Our Code
- Train the model
```bash
python main.py \
--name experiment_apada2seg \
--raw_A_dir ./preprocess/MRI_SEG/PROC/DCT/ \
--raw_A_seg_dir ./preprocess/MRI_SEG/PROC/DCT/ \
--raw_B_dir ./preprocess/MRI_SEG/PROC/MRI/ \
--sub_list_A ./preprocess/MRI_SEG/PROC/train_DCT.txt \
--sub_list_B ./preprocess/MRI_SEG/PROC/train_MRI.txt \
--batchSize 2 \
--angle 15 \
--model apada2seg_model_train \
--which_model_netS duseunet \
--pool_size 50 \
--no_dropout \
--apada2seg_run_model Train \
--dataset_mode apada2seg_train \
--input_nc 1  \
--output_nc 1 \
--output_nc_seg 2 \
--seg_norm DiceNorm \
--lambda_cc 0.1 \
--lambda_mind 0.1 \
--checkpoints_dir ./Checkpoints/MRI/ \
--display_id 0
```
where \
`--lambda_cc` defines the weights parameter for CC loss. \
`--lambda_mind`  defines the weights parameter for MIND loss. \
Other hyperparameters can be adjusted in the code as well.

- Test the model
```bash
python main.py \
--name experiment_apada2seg \
--raw_A_dir ./preprocess/MRI_SEG/PROC/DCT/ \
--raw_A_seg_dir ./preprocess/MRI_SEG/PROC/DCT/ \
--raw_B_dir ./preprocess/MRI_SEG/PROC/MRI/ \
--sub_list_A ./preprocess/MRI_SEG/PROC/train_DCT.txt \
--sub_list_B ./preprocess/MRI_SEG/PROC/train_MRI.txt \
--batchSize 1 \
--model apada2seg_model_test \
--which_model_netS duseunet \
--pool_size 50 \
--no_dropout \
--apada2seg_run_model TestSeg \
--dataset_mode apada2seg_test \
--input_nc 1  \
--seg_norm DiceNorm \
--output_nc 1 \
--output_nc_seg 2 \
--test_B_dir ./preprocess/MRI_SEG/PROC/MRI/ \
--test_img_list_file ./preprocess/MRI_SEG/PROC/test_MRI.txt \
--test_seg_output_dir ./Output/MRI/experiment_apada2seg \
--checkpoints_dir ./Checkpoints/MRI/ \
--which_epoch_S 10
```
Sample training/test scripts are provided under './scripts/' and can be directly executed.


### Registration based on APA2Seg-Net
Please refer to the Yale BioImage Suite (Command Line Version) for implementation of our registeration pipeline.
[[BIS Link]((https://bioimagesuiteweb.github.io/bisweb-manual/CommandLineTools.html)]


### Contact 
If you have any question, please file an issue or contact the author:
```
Bo Zhou: bo.zhou@yale.edu
```