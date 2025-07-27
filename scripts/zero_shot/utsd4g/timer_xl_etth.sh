#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

model_name=timer_xl
token_len=96
token_num=30
seq_len=$((token_num * token_len))

# checkpoint trained
CHECKPOINT_DIR=./checkpoints/forecast_utsd_timer_xl_Utsd_Npy_sl2880_it96_ot96_lr5e-05_bt512_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0
CHECKPOINT_NAME=checkpoint.pth

# ETT datasets
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2")

# forecast horizons
PRED_LENS=(96 192 336 720)

echo "forecast_era5_pretrain_timer_xl_Era5" >> result_long_term_forecast.txt

for DATASET in "${DATASETS[@]}"; do
  for pred_len in "${PRED_LENS[@]}"; do
    echo "======== Zero-shot on $DATASET | pred_len=$pred_len ========"

    echo "$DATASET - $pred_len" >> result_long_term_forecast.txt

    python -u run.py \
      --task_name forecast \
      --is_training 0 \
      --root_path ./dataset/ETT-small/ \
      --data_path $DATASET.csv \
      --model_id zero_shot_${DATASET}_pl$pred_len \
      --model $model_name \
      --data UnivariateDatasetBenchmark \
      --seq_len $seq_len \
      --input_token_len $token_len \
      --output_token_len $token_len \
      --test_seq_len $seq_len \
      --test_pred_len $pred_len \
      --e_layers 8 \
      --d_model 1024 \
      --d_ff 2048 \
      --batch_size 1024 \
      --learning_rate 0.00005 \
      --gpu 0 \
      --use_norm \
      --valid_last \
      --devices 0,1 \
      --num_workers 0 \
      --test_file_name $CHECKPOINT_NAME \
      --test_dir $(basename $CHECKPOINT_DIR) \
      --checkpoints $(dirname $CHECKPOINT_DIR) \
      --visualize
    echo ""
  done
done