import json

import cv2
import paddle
from paddle.io import Dataset


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

    def __getitem__(self, idx):
        label_map: dict = self.meta_info[idx]
        img = cv2.imread(self.dataset_dir + "/" + label_map["path"])
        label = label_map[self.color]
        img_tensor = paddle.to_tensor(img, dtype="float32").transpose(perm=[2,0,1])
        label_onehot = paddle.to_tensor([self.words_onehot[x] for x in label], dtype="int64")
        return img_tensor, label_onehot

    def __len__(self):
        return len(self.meta_info)
