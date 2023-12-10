#! -*- coding: utf-8 -*-

import json

import cv2
import numpy as np
from paddle.io import Dataset
from paddle.vision.transforms import transforms


class CaptchaDataset(Dataset):
    def __init__(self, dataset_dir: str, words_dict_path: str, mode: str = "train", color: str = "red"):
        super(CaptchaDataset, self).__init__()

        self.color = color
        self.dataset_dir = dataset_dir

        candidates = ["red", "blue", "black", "yellow"]
        assert color in candidates, f"color only can be one of {candidates}"

        assert mode in ["train", "test"], "mode can only be train or test!"
        json_file = dataset_dir + f"/{mode}.json"
        with open(json_file, 'r', encoding='utf-8') as fin:
            self.meta_info = json.load(fin)
        # 过滤掉颜色不存在的文件
        self.meta_info = [meta for meta in self.meta_info if meta.get(color)]

        with open(words_dict_path, 'r', encoding='utf-8') as fin:
            self.words_dict = fin.readlines()
        self.words_dict = [w.strip() for w in self.words_dict if w.strip()]
        self.words_onehot = {t: i for i, t in enumerate(self.words_dict)}

        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.55468268, 0.52301804, 0.51711713], std=[0.23353182, 0.23858833, 0.2390122]),
        ])

    def __getitem__(self, idx):
        label_map: dict = self.meta_info[idx]
        img = cv2.imread(self.dataset_dir + "/" + label_map["path"])
        label = label_map[self.color]
        img_arr = img.astype("float32").transpose([2, 0, 1]) / 255.0
        label_arr = np.zeros(shape=[1, len(self.words_onehot)], dtype="int64")
        label_arr[[self.words_onehot[x] for x in label]] = 1

        img_arr = self.transform(img_arr)
        return img_arr, label_arr

    def __len__(self):
        return len(self.meta_info)
