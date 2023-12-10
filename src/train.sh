#! /bin/bash

PROJ_DIR=$(dirname $(dirname $(readlink -f "$0")))
DATASET_DIR=${PROJ_DIR}/dataset/small

set -x
cd ${PROJ_DIR}/src
python train.py --dataset_dir ${DATASET_DIR} \
                --words_dict_path ${PROJ_DIR}/assets/words_dict.txt \
                --save_path ${PROJ_DIR}/output/checkpoint/model \
                --log_dir ${PROJ_DIR}/output/log
