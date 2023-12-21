#! -*- coding: utf-8 -*-

import json
import pathlib
import random
from typing import Optional

from PIL import Image
from paddle.io import Dataset

from util import DataUtil
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
        self.num_classes = len(self.vocabulary)

        self.data_util = DataUtil(vocabulary=self.vocabulary, max_len=max_len)
        assert channel in self.data_util.channels, f"channel only can be one of {self.data_util.channels}"
        assert auto_gen or dataset_dir, "dataset_dir must be set when auto_gen is False"
        assert mode in ["train", "test"], "mode can only be train or test!"

        if not auto_gen and dataset_dir:
            json_file = dataset_dir + f"/{mode}.json"
            with open(json_file, "r", encoding="utf-8") as fin:
                self.meta_info = json.load(fin)
            # 过滤掉颜色不存在的文件
            if channel not in ["text", "random"]:
                self.meta_info = [meta for meta in self.meta_info if meta.get(channel)]

    def _data_from(self, idx):
        """自动生成或者从本地数据读取"""
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

        # 如果是random则随机生成一个
        channel = random.choice(list(label_map)) if self.channel == "random" else self.channel
        return img, channel, label_map

    def __getitem__(self, idx):
        # 读取数据，来自本地或者自动生成
        img, channel, label_map = self._data_from(idx)

        # 图片加载&转换
        img_arr = self.data_util.process_img(img)
        # 颜色信息处理
        color_index = self.data_util.process_channel(channel)
        # 标签处理
        label_arr = self.data_util.process_label(label_map[channel])

        return (img_arr, color_index), label_arr

    def __len__(self):
        return len(self.meta_info) if not self.auto_gen else self.auto_num
