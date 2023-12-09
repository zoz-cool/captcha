set -x

PROJ_DIR = $(dirname $(dirname $(readlink -f "$0")))
DATASET_DIR = ${PROJ_DIR}/dataset/small

cd $PROJ_DIR && source venv/bin/activate
cd ${PROJ_DIR}/src
python train.py --dataset_dir ${DATASET_DIR} \
                --words_dict_path ${DATASET_DIR}/assets/words_dict.txt \
                --save_path ${DATASET_DIR}/output/checkpoint/model \
                --log_dir ${DATASET_DIR}/output/log
