#!/bin/bash

set -e

DATA_HOME_DIR=${DATA_HOME_DIR:-~/tf-data-multi}
INITIAL_STEPS=${INITIAL_STEPS:-15000}
EVAL_STEPS=${EVAL_STEPS:-5000}
CSV=${CSV:-posts_tags.csv}
PRETRAINED_CHECKPOINT_DIR=$DATA_HOME_DIR/checkpoints
MODEL_DIR=$DATA_HOME_DIR/models
DATASET_DIR=$DATA_HOME_DIR/dataset

for i in "$@"; do
  case $i in
    --data-dir=*)
    DATA_HOME_DIR="${i#*=}"
    shift
    ;;

    --checkpoint-dir=*)
    PRETRAINED_CHECKPOINT_DIR="${i#*=}"
    shift
    ;;

    --model-dir=*)
    MODEL_DIR="${i#*=}"
    shift
    ;;

    --dataset-dir=*)
    DATASET_DIR="${i#*=}"
    shift
    ;;

    *)
    echo "Unknown option: ${i}"
    exit 1
  esac
done

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
  tar -xvf inception_v4_2016_09_09.tar.gz
  mv inception_v4.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt
  rm inception_v4_2016_09_09.tar.gz
fi

# Run evaluation.
slimception/eval_image_classifier.py \
  --checkpoint_path=${MODEL_DIR} \
  --eval_dir=${MODEL_DIR} \
  --dataset_name=tags \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
  --multilabel=True
