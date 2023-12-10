#! -*- coding: utf-8 -*-

import os
import sys
import argparse

from datetime import datetime

import paddle
import numpy as np
from paddle.io import DataLoader
import visualdl as vdl
from paddle.static import InputSpec

import decoder
import model as m
import dataset


def train(args):
    writer = vdl.LogWriter(logdir=args.log_dir)

    # 获取训练数据
    train_dataset = dataset.CaptchaDataset(args.dataset_dir, args.words_dict_path, mode="train", color=args.channel)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    # 获取测试数据
    test_dataset = dataset.CaptchaDataset(args.dataset_dir, args.words_dict_path, mode="test", color=args.channel)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    # 获取模型
    num_classes = len(train_dataset.words_dict)
    model = m.Model(num_classes)
    img_size = train_dataset[0][0].shape
    paddle.summary(model, input_size=(args.batch_size, *img_size))

    # 设置优化方法
    boundaries = [10, 20, 50]
    lr = [0.1 ** val * args.lr for val in range(len(boundaries) + 1)]
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=boundaries, values=lr, verbose=False)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=scheduler)

    # 获取损失函数
    ctc_loss = paddle.nn.CTCLoss(blank=num_classes)

    # 加载预训练模型
    if args.pretrained:
        model.set_state_dict(paddle.load(os.path.join(args.pretrained, 'model.pdparams')))
        optimizer.set_state_dict(paddle.load(os.path.join(args.pretrained, 'optimizer.pdopt')))

    # 开始训练
    train_step = 0
    test_step = 0
    for epoch in range(args.num_epoch):
        for batch_id, (inputs, labels) in enumerate(train_loader()):
            out = model(inputs)
            out = paddle.transpose(out, perm=[1, 0, 2])
            input_lengths = paddle.full(shape=[out.shape[1]], fill_value=out.shape[0], dtype="int64")
            label_lengths = paddle.full(shape=[out.shape[1]], fill_value=4, dtype="int64")
            # 计算损失
            loss = ctc_loss(out, labels, input_lengths, label_lengths)
            loss.backward()

            optimizer.step()
            optimizer.clear_grad()
            # 多卡训练只使用一个进程打印
            if batch_id % 100 == 0:
                print('[%s] Train epoch %d, batch %d, loss: %f' % (datetime.now(), epoch, batch_id, loss))
                writer.add_scalar('Train loss', loss, train_step)
                train_step += 1
        if (epoch % 10 == 0 and epoch != 0) or epoch == args.num_epoch - 1:
            # 执行评估
            model.eval()
            cer = evaluate(model, test_loader, train_dataset.words_dict)
            print('[%s] Test epoch %d, cer: %f' % (datetime.now(), epoch, cer))
            writer.add_scalar('Test cer', cer, test_step)
            test_step += 1
            model.train()
        # 记录学习率
        writer.add_scalar('Learning rate', scheduler.last_lr, epoch)
        scheduler.step()
        # 保存模型
        paddle.jit.save(layer=model, path=args.save_path,
                        input_spec=[InputSpec(shape=[None, *img_size], dtype='float32')])


# 评估模型
def evaluate(model, test_loader, words_dict):
    cer_result = []
    for batch_id, (inputs, labels) in enumerate(test_loader()):
        # 执行识别
        outs = model(inputs)
        outs = paddle.nn.functional.softmax(outs)
        # 解码获取识别结果
        label_list = []
        out_strings = []
        for out in outs:
            out_string = decoder.ctc_greedy_decoder(out, words_dict)
            out_strings.append(out_string)
        for label in labels:
            labels = decoder.label_to_string(label, words_dict)
            label_list.append(labels)
        for out_string, label in zip(*(out_strings, label_list)):
            print(label, out_string)
            # 计算字错率
            c = decoder.cer(out_string, label) / float(len(label))
            cer_result.append(c)
    cer_result = float(np.mean(cer_result))
    return cer_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/home/aistudio/data/data251502/small")
    parser.add_argument("--words_dict_path", type=str, default="/home/aistudio/work/words_dict.txt")
    parser.add_argument("--save_path", type=str, default="/home/aistudio/work/output/checkpoint/model")
    parser.add_argument("--log_dir", type=str, default="/home/aistudio/work/output/log")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--channel", type=str, default="red")

    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    arg = parse_args()
    train(arg)
