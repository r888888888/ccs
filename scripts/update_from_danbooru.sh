#!/bin/bash

set -e

TF_CPP_MIN_LOG_LEVEL=2
INITIAL_STEPS=40000
EVAL_STEPS=5000
CSV=posts_chars.csv
DATA_HOME_DIR=/var/lib/ccs/data
PRETRAINED_CHECKPOINT_DIR=$DATA_HOME_DIR/checkpoints
MODEL_DIR=$DATA_HOME_DIR/models
DATASET_DIR=$DATA_HOME_DIR/dataset

wget https://isshiki.donmai.us/exports/posts_chars.csv -O /var/lib/ccs/data/dataset/posts_chars.csv

nvidia-docker stop $(nvidia-docker ps -aq) || true
nvidia-docker rm $(nvidia-docker ps -aq) || true

rm -rf $DATA_HOME_DIR/dataset/*.txt
rm -rf $DATA_HOME_DIR/dataset/*.tfrecord
rm -rf $DATA_HOME_DIR/models/*

slimception/download_and_convert_data.py \
  --dataset_dir=${DATASET_DIR} \
  --num_classes_file=num_char_classes.txt \
  --num_images_file=num_char_images.txt \
  --dataset_name=characters \
  --source_csv=${CSV} \
  --min_term_df=150

slimception/train_image_classifier.py \
  --train_dir=${MODEL_DIR} \
  --dataset_name=characters \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt \
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
  --max_number_of_steps=${INITIAL_STEPS} \
  --batch_size=32 \
  --learning_rate=0.04 \
  --learning_rate_decay_type=exponential \
  --save_interval_secs=300 \
  --save_summaries_secs=1800 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --weight_decay=0.00004 \
  --multilabel=False

slimception/train_image_classifier.py \
  --train_dir=${MODEL_DIR}/all \
  --dataset_name=characters \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
  --checkpoint_path=${MODEL_DIR} \
  --max_number_of_steps=${EVAL_STEPS} \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=1800 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --weight_decay=0.00004 \
  --multilabel=False

nvidia-docker run -d -p 5000:5000 -v /etc/ccs:/etc/ccs -v /var/lib/ccs/data:/var/lib/ccs/data ccs:latest
