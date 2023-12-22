#! /bin/bash

PROJ_DIR=$(dirname $(dirname $(readlink -f "$0")))

# 指定训练参数
# 默认情况下可以不指定数据集路径，采用自动生成策略
DATASET_DIR=
# 自动生成时数据集大小（实际上每个epoch生成的数据集都是不同的）
AUTO_NUM=1000000
# 可以指定预训练模型来实现增量训练
PRETRAINED=
# 评测评率和保存频率
EVAL_FREQ=5
SAVE_FREQ=1
# 生成的验证码最大长度
MAX_LEN=6
# 模型是识别具体的某个颜色还是识别所有颜色
CHANNEL=random
# 是否采用简单模式，简单模式下验证码没有汉字
SIMPLE_MODE=
# 采用哪个模型作为特征提取网络
MODEL="custom"
# 批次大小
BATCH_SIZE=1000
# 训练轮数
NUM_EPOCH=100
# 学习率
LR=0.001
# 多进程读取数据
NUM_WORKERS=20
# 是否使用wandb来可视化(指定online或offline，不用wandb可视化则不填写)
WANDB_MODE=offline
WANDB_NAME=
# 用几卡训练
GPUS=0,1,2,3

if [[ ! -z $DATASET_DIR ]]; then
    DATASET_DIR="--dataset_dir $DATASET_DIR"
fi

if [[ ! -z $PRETRAINED ]]; then
    PRETRAINED="--pretrained $PRETRAINED"
fi

if [[ ! -z $SIMPLE_MODE ]]; then
    SIMPLE_MODE="--simple_mode"
fi

if [[ ! -z $WANDB_MODE ]]; then
    WANDB_MODE="--wandb_mode $WANDB_MODE"
fi

if [[ ! -z $WANDB_NAME ]]; then
    WANDB_NAME="--wandb_name $WANDB_NAME"
fi

if [[ ${#GPUS} -gt 1 ]]; then
    DISTRIBUTED_LAUNCH="-m paddle.distributed.launch"
fi
# 运行train.py文件
CUDA_VISIBLE_DEVICES=$GPUS python $DISTRIBUTED_LAUNCH train.py \
                --auto_num $AUTO_NUM \
                --eval_freq $EVAL_FREQ \
                --save_freq $SAVE_FREQ \
                --max_len $MAX_LEN \
                --channel $CHANNEL \
                --model $MODEL \
                --batch_size $BATCH_SIZE \
                --num_epoch $NUM_EPOCH \
                --lr $LR \
                --num_workers $NUM_WORKERS \
                $WANDB_NAME $WANDB_MODE $SIMPLE_MODE $DATASET_DIR $PRETRAINED
