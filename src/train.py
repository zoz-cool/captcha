#! -*- coding: utf-8 -*-

import os
import re
import sys
import random
import argparse
import pathlib

from datetime import datetime

import paddle
import numpy as np
import prettytable
from paddle.io import DataLoader
import visualdl as vdl
from paddle.static import InputSpec

import decoder
import model
import dataset


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.train_steps = 0
        self.test_steps = 0

        self._init_data()
        self._init_model()

    def _init_data(self):
        self.writer = vdl.LogWriter(logdir=self.args.log_dir)
        # 获取训练数据
        self.train_dataset = dataset.CaptchaDataset(self.args.dataset_dir, self.args.vocabulary_path, mode="train",
                                                    color=self.args.channel)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

        # 获取测试数据
        self.test_dataset = dataset.CaptchaDataset(self.args.dataset_dir, self.args.vocabulary_path, mode="test",
                                                   color=self.args.channel)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.args.batch_size)

        self.num_classes = len(self.train_dataset.vocabulary)

        t = prettytable.PrettyTable(["field", "number"])
        t.add_row(["num_classes", self.num_classes])
        t.add_row(["train_dataset", len(self.train_dataset)])
        t.add_row(["test_dataset", len(self.test_dataset)])
        print(t)

    def _init_model(self):
        # 获取模型
        self.model = model.Model(self.num_classes, self.args.max_len)
        self.img_size = self.train_dataset[0][0].shape

        # 打印模型和数据信息
        paddle.summary(self.model, input_size=(self.args.batch_size, *self.img_size))

        # 设置优化方法
        self.scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=self.args.lr, factor=0.1, patience=10)
        self.optimizer = paddle.optimizer.Adam(parameters=self.model.parameters(), learning_rate=self.scheduler)
        # 获取损失函数
        self.ctc_loss = paddle.nn.CTCLoss(blank=self.num_classes)

        # 加载预训练模型
        if self.args.pretrained:
            print(f"load pretrained model from {self.args.pretrained}")
            self.model.set_state_dict(paddle.load(os.path.join(self.args.pretrained, 'model.pdparams')))
            self.optimizer.set_state_dict(paddle.load(os.path.join(self.args.pretrained, 'optimizer.pdopt')))

    def train_step(self, epoch_id, batch_id, input_data):
        inputs, labels = input_data
        out = self.model(inputs)
        input_lengths = paddle.full(shape=[out.shape[1]], fill_value=out.shape[0], dtype="int64")
        label_lengths = paddle.sum(labels != -1, axis=-1, dtype="int64")
        # 计算损失
        loss = self.ctc_loss(out, labels, input_lengths, label_lengths)
        loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()
        # 打印日志
        if batch_id % 100 == 0:
            print('[%s] Train epoch %d, batch %d, lr %f, loss: %f' % (
                datetime.now(), epoch_id, batch_id, self.scheduler.last_lr, float(loss)))
            self.writer.add_scalar('Train loss', float(loss), self.train_steps)
            self.train_steps += 1
        # 记录学习率
        self.writer.add_scalar('Learning rate', self.scheduler.last_lr, epoch_id)
        return float(loss)

    def test_epoch(self, epoch_id):
        if (epoch_id % self.args.eval_per_epoch == 0 and epoch_id != 0) or epoch_id == self.args.num_epoch - 1:
            # 执行评估
            cer = self.evaluate()
            print('[%s] Test epoch %d, cer: %f' % (datetime.now(), epoch_id, cer))
            self.writer.add_scalar('Test cer', cer, self.test_steps)
            self.test_steps += 1

    def save_epoch(self, epoch_id):
        """保存模型"""
        if (epoch_id % self.args.save_per_epoch == 0 and epoch_id != 0) or (epoch_id == self.args.num_epoch - 1) or (
                epoch_id == 0 and self.args.save_per_epoch == 1):
            os.makedirs(self.args.save_path, exist_ok=True)
            # 移除超出限制的模型
            epoch_dirs = [e_dir for e_dir in os.listdir(self.args.save_path) if re.match(r"^e\d+$", e_dir)]
            if len(epoch_dirs) >= self.args.max_keep:
                epoch_dirs = sorted(epoch_dirs, key=lambda e_dir: int(e_dir[1:]))
                for epoch_dir in epoch_dirs[:len(epoch_dirs) - self.args.max_keep + 1]:
                    os.system(f"rm -rf {str(self.args.save_path)}/{epoch_dir}")
            # 保存模型
            print('[%s] Save model on epoch %d' % (datetime.now(), epoch_id))
            save_epoch_path = "/".join([self.args.save_path, f"e{epoch_id}", "model"])
            paddle.jit.save(layer=self.model, path=save_epoch_path,
                            input_spec=[InputSpec(shape=[None, *self.img_size], dtype='float32')])

    def train_epoch(self, epoch_id):
        """训练一个epoch"""
        loss_list = []
        for batch_id, input_data in enumerate(self.train_loader()):
            loss = self.train_step(epoch_id, batch_id, input_data)
            loss_list.append(loss)
        
        self.scheduler.step(np.mean(loss_list), epoch_id)

    def train(self):
        """开始训练"""
        if os.path.exists(self.args.save_path):
            os.system(f"rm -rf {self.args.save_path}")
        for epoch_id in range(self.args.num_epoch):
            self.train_epoch(epoch_id)
            self.test_epoch(epoch_id)
            self.save_epoch(epoch_id)

    def evaluate(self):
        """评估模型"""
        self.model.eval()
        cer_result = []
        samples = []
        for batch_id, (inputs, labels) in enumerate(self.test_loader()):
            # 执行识别
            outs = self.model(inputs)
            outs = paddle.transpose(outs, perm=[1, 0, 2])
            outs = paddle.nn.functional.softmax(outs)
            # 解码获取识别结果
            truth_list = []
            pred_list = []
            for out in outs:
                pred = decoder.ctc_greedy_decoder(out, self.test_dataset.vocabulary)
                pred_list.append(pred)
            for label in labels:
                label_text = decoder.label_to_string(label, self.test_dataset.vocabulary)
                truth_list.append(label_text)
            idx = random.choice(range(len(truth_list)))
            samples.append((truth_list[idx], pred_list[idx]))
            for pred, truth in zip(*(pred_list, truth_list)):
                # 计算字错率
                c = decoder.cer(pred, truth) / float(len(truth))
                cer_result.append(c)
        print("random.sample:", random.sample(samples, 10))
        cer_result = float(np.mean(cer_result))
        self.model.train()
        return cer_result


def parse_args():
    proj_dir = pathlib.Path(__file__).parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=str(proj_dir / "dataset"))
    parser.add_argument("--vocabulary_path", type=str, default=str(proj_dir / "assets/vocabulary.txt"))
    parser.add_argument("--save_path", type=str, default=str(proj_dir / "output/checkpoint"))
    parser.add_argument("--log_dir", type=str, default=str(proj_dir / "output/log"))

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--channel", type=str, default="red")
    parser.add_argument("--eval_per_epoch", type=int, default=2)
    parser.add_argument("--save_per_epoch", type=int, default=1)
    parser.add_argument("--max_keep", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=6)

    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    arg = parse_args()
    Trainer(arg).train()
