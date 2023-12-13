#! -*- coding: utf-8 -*-

import os
import sys
import argparse
import pathlib
import shutil

import paddle
import prettytable
import visualdl as vdl

import model
import dataset
import metric
import loss


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.train_steps = 0
        self.test_steps = 0

        self._init_data()
        self._init_model()

    def _init_data(self):
        self.writer = vdl.LogWriter(logdir=self.args.log_dir)
        # 传入数据集地址时使用已有数据集，否则边训练边生成数据集
        # 获取训练数据
        self.train_dataset = dataset.CaptchaDataset(
            vocabulary_path=self.args.vocabulary_path,
            dataset_dir=self.args.dataset_dir,
            auto_gen=self.args.auto_num > 0 and self.args.dataset_dir is None,
            auto_num=self.args.auto_num,
            mode="train",
            channel=self.args.channel,
            max_len=self.args.max_len,
            simple_mode=self.args.simple_mode
        )

        # 获取测试数据
        self.test_dataset = dataset.CaptchaDataset(
            vocabulary_path=self.args.vocabulary_path,
            dataset_dir=self.args.dataset_dir,
            auto_gen=self.args.auto_num > 0 and self.args.dataset_dir is None,
            auto_num=min(self.args.auto_num // 2, 100_000),  # 自动生成时测试集数量为训练集的一半，同时限制不超过10w
            mode="test",
            channel=self.args.channel,
            max_len=self.args.max_len,
            simple_mode=self.args.simple_mode
        )

        self.vocabulary = self.train_dataset.vocabulary
        self.num_classes = len(self.train_dataset.vocabulary)

        t = prettytable.PrettyTable(["field", "number"])
        t.add_row(["num_classes", self.num_classes])
        t.add_row(["train_dataset", len(self.train_dataset)])
        t.add_row(["test_dataset", len(self.test_dataset)])
        print(t)

    def _init_model(self):
        # 获取模型
        m = model.Model(self.num_classes, self.args.max_len)
        img_size = self.train_dataset[0][0].shape
        label_size = self.train_dataset[0][1].shape
        print("img_size", img_size)
        print("label-size", label_size)
        inputs_shape = paddle.static.InputSpec([-1, *img_size], dtype='float32', name='input')
        labels_shape = paddle.static.InputSpec([-1, *label_size], dtype='int64', name='label')
        self.model = paddle.Model(m, inputs_shape, labels_shape)

        # 打印模型和数据信息
        self.model.summary()

        # 设置优化方法
        # scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=self.args.lr, factor=0.1, patience=10)
        self.optimizer = paddle.optimizer.Adam(parameters=self.model.parameters(), learning_rate=self.args.lr)
        # 获取损失函数
        ctc_loss = loss.CTCLoss(self.num_classes)

        self.model.prepare(self.optimizer, ctc_loss, metric.WordsPrecision(self.vocabulary))

        # 加载预训练模型
        if self.args.pretrained:
            print(f"Load pretrained model from {self.args.pretrained}")
            self.model.load(self.args.pretrained)

    def train(self):
        """开始训练"""
        if os.path.exists(self.args.save_dir):
            shutil.rmtree(self.args.save_dir)
        self.model.fit(train_data=self.train_dataset, eval_data=self.test_dataset, batch_size=self.args.batch_size,
                       shuffle=True, epochs=self.args.num_epoch,
                       eval_freq=self.args.eval_freq, log_freq=10, save_freq=self.args.save_freq,
                       save_dir=self.args.save_dir, num_workers=0, verbose=1)
        self.model.save(self.args.save_dir + "/inference/model", False)  # save for inference


def parse_args():
    proj_dir = pathlib.Path(__file__).absolute().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--vocabulary_path", type=str, default=str(proj_dir / "assets/vocabulary.txt"))
    parser.add_argument("--save_dir", type=str, default=str(proj_dir / "output/checkpoint"))
    parser.add_argument("--log_dir", type=str, default=str(proj_dir / "output/log"))

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--channel", type=str, default="red")
    parser.add_argument("--eval_freq", type=int, default=2)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--max_keep", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=6)
    parser.add_argument("--auto_num", type=int, default=100_000)
    parser.add_argument("--simple_mode", action="store_true")

    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    arg = parse_args()
    Trainer(arg).train()
