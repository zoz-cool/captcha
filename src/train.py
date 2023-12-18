#! -*- coding: utf-8 -*-

import sys
import argparse
import pathlib

import paddle
import prettytable

import model
import dataset
import metric
import loss


class PrintLastLROnEpochEnd(paddle.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}, learning rate is {self.model._optimizer.get_lr()}")


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self._init_data()
        self._init_model()

    def _init_data(self):
        # 传入数据集地址时使用已有数据集，否则边训练边生成数据集
        # 获取训练数据
        auto_gen = self.args.dataset_dir is None
        self.train_dataset = dataset.CaptchaDataset(
            vocabulary_path=self.args.vocabulary_path,
            dataset_dir=self.args.dataset_dir,
            auto_gen=auto_gen,
            auto_num=self.args.auto_num,
            mode="train",
            channel=self.args.channel,
            max_len=self.args.max_len,
            simple_mode=self.args.simple_mode
        )
        self.train_dataloader = paddle.io.DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                                     num_workers=self.args.num_workers, use_shared_memory=False)

        # 获取测试数据
        self.test_dataset = dataset.CaptchaDataset(
            vocabulary_path=self.args.vocabulary_path,
            dataset_dir=self.args.dataset_dir,
            auto_gen=auto_gen,
            auto_num=min(self.args.auto_num // 2, 100_000),  # 自动生成时测试集数量为训练集的一半，同时限制不超过10w
            mode="test",
            channel=self.args.channel,
            max_len=self.args.max_len,
            simple_mode=self.args.simple_mode
        )
        self.test_dataloader = paddle.io.DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False,
                                                    num_workers=self.args.num_workers, use_shared_memory=False)

        self.vocabulary = self.train_dataset.vocabulary
        self.num_classes = len(self.train_dataset.vocabulary)

        t = prettytable.PrettyTable(["field", "number"])
        t.add_row(["num_classes", self.num_classes])
        t.add_row(["train_dataset", len(self.train_dataset)])
        t.add_row(["test_dataset", len(self.test_dataset)])
        print(t)

    def _init_model(self):
        # 获取模型
        print("Use model {}".format(self.args.model))
        m = model.Model(self.num_classes, self.args.max_len, feature_net=self.args.model)
        img_size = self.train_dataset[0][0][0].shape
        label_size = self.train_dataset[0][1].shape
        inputs_shape = paddle.static.InputSpec([None, *img_size], dtype='float32', name='input')
        color_shape = paddle.static.InputSpec([1, ], dtype='float32', name='color')
        labels_shape = paddle.static.InputSpec([None, *label_size], dtype='int64', name='label')
        self.model = paddle.Model(m, (inputs_shape, color_shape), labels_shape)

        # 打印模型和数据信息
        self.model.summary(input_size=([self.args.batch_size, *img_size], [1]))

        # 设置优化方法
        def make_optimizer(parameters=None):
            boundaries = [5, 50, 100]
            warmup_steps = 4
            values = [self.args.lr * (0.1 ** i) for i in range(len(boundaries) + 1)]
            learning_rate = paddle.optimizer.lr.PiecewiseDecay(
                boundaries=boundaries, values=values)
            learning_rate = paddle.optimizer.lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                start_lr=self.args.lr / 5.,
                end_lr=self.args.lr,
                verbose=False)
            optimizer = paddle.optimizer.Adam(
                learning_rate=learning_rate,
                parameters=parameters)
            return optimizer

        self.optimizer = make_optimizer(self.model.parameters())
        # 获取损失函数
        ctc_loss = loss.CTCLoss(self.num_classes)

        self.model.prepare(self.optimizer, ctc_loss, metrics=[metric.WordsErrorRate(self.vocabulary)])

        # 加载预训练模型
        if self.args.pretrained:
            print(f"Load pretrained model from {self.args.pretrained}")
            self.model.load(self.args.pretrained)

    def train(self):
        """开始训练"""
        vdl_log_dir = str(pathlib.Path(self.args.log_dir, "vdl"))
        callbacks = [paddle.callbacks.VisualDL(log_dir=vdl_log_dir),
                     paddle.callbacks.LRScheduler(by_step=False, by_epoch=True),
                     PrintLastLROnEpochEnd()]
        if self.args.wandb_mode in ["online", "offline"]:
            name = f"{self.args.model}-bs{self.args.batch_size}"
            if self.args.wandb_name:
                name = name + "-" + self.args.wandb_name
            print(f"Use wandb to record log, name: {name}, mode: {self.args.wandb_mode}")
            wandb_callback = paddle.callbacks.WandbCallback(project="captcha",
                                                            dir=self.args.log_dir,
                                                            name=name,
                                                            mode=self.args.wandb_mode,
                                                            job_type="simple" if self.args.simple_mode else "complex",
                                                            group=self.args.channel)
            callbacks.append(wandb_callback)
        self.model.fit(train_data=self.train_dataloader, eval_data=self.test_dataloader, epochs=self.args.num_epoch,
                       callbacks=callbacks, eval_freq=self.args.eval_freq, log_freq=10, save_freq=self.args.save_freq,
                       save_dir=self.args.save_dir, verbose=1)
        self.model.save(self.args.save_dir + "/inference/model", False)  # save for inference


def parse_args():
    proj_dir = pathlib.Path(__file__).absolute().parent.parent

    parser = argparse.ArgumentParser()
    # 初始化参数
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--vocabulary_path", type=str, default=str(proj_dir / "assets/vocabulary.txt"))
    parser.add_argument("--save_dir", type=str, default=str(proj_dir / "output/checkpoint"))
    parser.add_argument("--log_dir", type=str, default=str(proj_dir / "output/log"))
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="")
    parser.add_argument("--auto_num", type=int, default=10000)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--eval_freq", type=int, default=2)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=6, help="生成的验证码最大长度")
    # 模型参数
    parser.add_argument("--channel", type=str, default="text")
    parser.add_argument("--simple_mode", action="store_true")
    parser.add_argument("--model", type=str, default="custom")
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    arg = parse_args()
    Trainer(arg).train()
