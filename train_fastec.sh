#!/bin/bash

# !! Updata the path to the dataset and directory to 
# !! save your trained models with your own local path !!
fastec_dataset_type=Fastec
fastec_root_path_training_data=/data/local_userdata/fanbin/raw_data/faster/data_train/train/

log_dir=/home/fanbin/fan/RSC/JAMNet/deep_unroll_weights/
#
cd deep_unroll_net


python train_JAM.py \
          --dataset_type=$fastec_dataset_type \
          --dataset_root_dir=$fastec_root_path_training_data \
          --log_dir=$log_dir \
          --lamda_L1=10 \
          --lamda_perceptual=1 \
          --lamda_flow_smoothness=0.1 \
          --crop_sz_H=480 \
          #--continue_train=True \
          #--start_epoch=201 \
          #--model_label=200


