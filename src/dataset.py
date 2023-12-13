#! -*- coding: utf-8 -*-

import json
import pathlib
import random
from typing import Optional

import numpy as np
from PIL import Image
from paddle.io import Dataset
from paddle.vision.transforms import transforms

from preprocess.captcha_generator import CaptchaGenerator

assets_dir = pathlib.Path(__file__).absolute().parent.parent / "assets"


class CaptchaDataset(Dataset):
    """数据集加载器
    auto_gen: 是否使用自动生成，当此参数启用时，数据即时生成，否则从本地路径中加载数据
    auto_num: 当使用auto_gen时数据可以是无限多，但为了训练适配训练流程，需指定数据集大小
    dataset_dir: 当从本地路径加载数据集时的路径地址
    channel: 生成的标签需要的颜色类型，可选两种模式，第一种是指定具体颜色类型，第二种是随机选择一个颜色类型，这种模式下模型能识别所有颜色类型
    """

    def __init__(
            self,
            vocabulary_path: Optional[str] = str(assets_dir / "vocabulary.txt"),
            auto_gen: bool = False,
            auto_num: int = 100_000,
            dataset_dir: Optional[str] = None,
            mode: str = "train",
            channel: str = "text",
            max_len: int = 6,
            simple_mode: bool = False
    ):
        super(CaptchaDataset, self).__init__()

        self.auto_gen = auto_gen
        self.auto_num = auto_num
        self.max_len = max_len
        self.channel = channel
        self.dataset_dir = dataset_dir
        self.generator = CaptchaGenerator(vocabulary_path=vocabulary_path, max_words=max_len, simple_mode=simple_mode)
        self.vocabulary = self.generator.characters if simple_mode else self.generator.vocabulary
        self._vocabulary_dict = {t: i for i, t in enumerate(self.vocabulary)}

        self.channels = ["red", "blue", "black", "yellow", "text", "random"]
        assert channel in self.channels, f"channel only can be one of {self.channels}"
        assert auto_gen or dataset_dir, "dataset_dir must be set when auto_gen is False"
        assert mode in ["train", "test"], "mode can only be train or test!"

        if not auto_gen and dataset_dir:
            json_file = dataset_dir + f"/{mode}.json"
            with open(json_file, "r", encoding="utf-8") as fin:
                self.meta_info = json.load(fin)
            # 过滤掉颜色不存在的文件
            if channel not in ["text", "random"]:
                self.meta_info = [meta for meta in self.meta_info if meta.get(channel)]

        self.std = np.array([0.23375643, 0.23862716, 0.23951546])
        self.mean = np.array([0.55456273, 0.5225813, 0.51677391])
        self.transform = transforms.Compose([transforms.Normalize(mean=self.mean, std=self.std)])

    def restore(self, img_arr):
        """img_arr恢复为图片对象"""
        img = img_arr.transpose([1, 2, 0])
        norm = (img * self.std + self.mean) * 255.0
        return norm.astype(np.uint8)

    def __getitem__(self, idx):
        if self.auto_gen:  # 自动生成
            img, label_map = self.generator.gen_one(min_num=4, max_num=self.max_len)
            if self.channel not in ["text", "random"]:
                while self.channel not in label_map:  # 生成的图片无对应颜色数据则重新生成
                    img, label_map = self.generator.gen_one(min_num=4, max_num=self.max_len)
        elif self.dataset_dir:  # 从本地数据读取
            label_map: dict = self.meta_info[idx]
            img = Image.open(self.dataset_dir + "/" + label_map["path"])
        else:
            raise ValueError("dataset_dir must be set when auto_gen is False")

        # 图片加载&转换
        img = img.convert("RGB")
        img_arr = np.array(img, np.float32).transpose([2, 0, 1]) / 255.0

        # 标签处理
        # 如果是random则随机生成一个
        channel = random.choice(list(label_map)) if self.channel == "random" else self.channel
        label = label_map[channel]
        label_seq = [self._vocabulary_dict[t] for t in label]
        if len(label_seq) < self.max_len:
            label_seq += [-1] * (self.max_len - len(label_seq))
        label_arr = np.array(label_seq).astype("int32")

        # 将图像通道和标签颜色索引拼接在一起（这么使用只是为了满足paddle高级接口传入inputs只接受一个参数）
        color_index = self.channels.index(channel)
        color_arr = color_index * np.ones([1, *img_arr.shape[1:]], dtype=np.float32) / len(self.channels)
        img_arr = np.concatenate([img_arr, color_arr]).astype(np.float32)

        return img_arr, label_arr

    def __len__(self):
        return len(self.meta_info) if not self.auto_gen else self.auto_num
