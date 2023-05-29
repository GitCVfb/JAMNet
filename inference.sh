# !! Updata the path to the dataset and directory to 
# !! load the trained models with your own local path !!
carla_dataset_type=Carla
carla_root_path_test_data=/data/local_userdata/fanbin/raw_data/carla/data_test/test/

fastec_dataset_type=Fastec
fastec_root_path_test_data=/data/local_userdata/fanbin/raw_data/faster/data_test/test/

bsrsc_dataset_type=BSRSC
bsrsc_root_path_test_data=/data/local_userdata/fanbin/raw_data/BSRSC/test/

results_dir=/home/fanbin/fan/RSC/JAMNet/deep_unroll_results/



model_dir1=../deep_unroll_weights/model_weights/carla/
model_dir2=../deep_unroll_weights/model_weights/fastec/
model_dir3=../deep_unroll_weights/model_weights/bsrsc/

cd deep_unroll_net


python inference.py \
          --dataset_type=$bsrsc_dataset_type \
          --dataset_root_dir=$bsrsc_root_path_test_data \
          --log_dir=$model_dir3 \
          --results_dir=$results_dir \
          --crop_sz_H=768 \
          --compute_metrics \
          --model_label=pre

:<<!
python inference.py \
          --dataset_type=$fastec_dataset_type \
          --dataset_root_dir=$fastec_root_path_test_data \
          --log_dir=$model_dir2 \
          --results_dir=$results_dir \
          --crop_sz_H=480 \
          --compute_metrics \
          --model_label=pre

python inference.py \
          --dataset_type=$carla_dataset_type \
          --dataset_root_dir=$carla_root_path_test_data \
          --log_dir=$model_dir1 \
          --results_dir=$results_dir \
          --crop_sz_H=448 \
          --compute_metrics \
          --model_label=pre
!




