#! /bin/bash

PROJ_DIR=$(dirname $(dirname $(readlink -f "$0")))
DATASET_DIR=${PROJ_DIR}/dataset_simple36

BATCH_SIZE=200
NUM_EPOCH=100
LR=0.01
PRETRAINED=
CHANNEL=red
EVAL_PER_EPOCH=2

cd ${PROJ_DIR}/src

if [[ ! -z ${PRETRAINED} ]]
then
    PRETRAINED="--pretrained ${PRETRAINED}"
fi

set -x
python train.py --dataset_dir ${DATASET_DIR} \
                --vocabulary_path ${PROJ_DIR}/assets/vocabulary.txt \
                --save_path ${PROJ_DIR}/output/checkpoint \
                --log_dir ${PROJ_DIR}/output/log \
                --batch_size ${BATCH_SIZE} \
                --num_epoch ${NUM_EPOCH} \
                --lr ${LR} \
                --channel ${CHANNEL} \
                --eval_per_epoch ${EVAL_PER_EPOCH} \
                ${PRETRAINED}
