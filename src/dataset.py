#! -*- coding: utf-8 -*-

import json

import cv2
import numpy as np
from paddle.io import Dataset
from paddle.vision.transforms import transforms


class CaptchaDataset(Dataset):
    def __init__(self, dataset_dir: str, vocabulary_ath: str, mode: str = "train", color: str = "red", max_len=6):
        super(CaptchaDataset, self).__init__()

        self.max_len = max_len
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

        with open(vocabulary_ath, 'r', encoding='utf-8') as fin:
            self.vocabulary = fin.readlines()
        self.vocabulary = [w.strip() for w in self.vocabulary if w.strip()]
        self._vocabulary_dict = {t: i for i, t in enumerate(self.vocabulary)}

        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.55456273, 0.5225813,  0.51677391], std=[0.23375643, 0.23862716, 0.23951546]),
        ])


    def __getitem__(self, idx):
        label_map: dict = self.meta_info[idx]
        img = cv2.imread(self.dataset_dir + "/" + label_map["path"])
        label = label_map[self.color]
        img_arr = img.astype("float32").transpose([2, 0, 1]) / 255.0
        img_arr = self.transform(img_arr)

        label_seq = [self._vocabulary_dict[t] for t in label]
        if len(label_seq) < self.max_len:
            label_seq += [-1] * (self.max_len - len(label_seq))
        label_arr = np.array(label_seq).astype("int32")
        
        return img_arr, label_arr

    def __len__(self):
        return len(self.meta_info)
