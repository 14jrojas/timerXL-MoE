#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

model_name=timer_xl_moe
token_len=96
token_num=30
seq_len=$((token_num * token_len))

CHECKPOINT_DIR=./checkpoints/forecast_utsd_timer_xl_moe_Utsd_Npy_sl2880_it96_ot96_lr5e-05_bt512_wd0_el8_dm512_dff512_nh8_cosTrue_test_0
CHECKPOINT_NAME=checkpoint.pth

PRED_LENS=(96 192 336 720)

for pred_len in "${PRED_LENS[@]}"; do
  echo "======== Zero-shot on ECL | pred_len=$pred_len ========"

  echo "ECL - $pred_len" >> result_long_term_forecast.txt

  python -u run.py \
    --task_name forecast \
    --is_training 0 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id zero_shot_ECL_pl$pred_len \
    --model $model_name \
    --data UnivariateDatasetBenchmark \
    --seq_len $seq_len \
    --input_token_len $token_len \
    --output_token_len $token_len \
    --test_seq_len $seq_len \
    --test_pred_len $pred_len \
    --e_layers 8 \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 512 \
    --learning_rate 0.00005 \
    --gpu 0 \
    --use_norm \
    --valid_last \
    --devices 0 \
    --num_workers 0 \
    --test_file_name $CHECKPOINT_NAME \
    --test_dir $(basename $CHECKPOINT_DIR) \
    --checkpoints $(dirname $CHECKPOINT_DIR) \
    --visualize
done
