#!/bin/bash

# create an empty folder for experimental results
mkdir -p experiments/results_demo_fastec
mkdir -p experiments/results_demo_carla
mkdir -p experiments/results_demo_bsrsc

cd deep_unroll_net


python inference_demo.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_bsrsc \
            --data_dir='../demo/BSRSC' \
            --log_dir=../deep_unroll_weights/model_weights/bsrsc

python inference_demo.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_carla \
            --data_dir='../demo/Carla' \
            --log_dir=../deep_unroll_weights/model_weights/carla

python inference_demo.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_fastec \
            --data_dir='../demo/Fastec' \
            --log_dir=../deep_unroll_weights/model_weights/fastec
