"""
数据集加载器
"""
import re
import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

from .config import Config

config = Config()
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                transforms.Grayscale(num_output_channels=1)])


class CaptchaSet(Dataset):
    """
    数据集获取，可返回一个数据对象，文件命名示例：dataset/train/A2S_001.png
    """

    def __init__(self, root, transform):
        self.transform = transform
        self.images = glob.glob(os.path.join(root, "*.png"))

    def __getitem__(self, index):
        file = self.images[index]
        labels_str = re.split(r'[._-]', os.path.basename(file))[0]
        labels_list = [config.classes[c] for c in labels_str]
        image = Image.open(file)
        if self.transform:
            image = self.transform(image)
        return image, torch.IntTensor(labels_list)

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    """
    自定义数据集加载方式：将标签转为维度一致的数组，
    一个批次中的数据标签会被展平为一维，通过seq_lengths记录每个样本标签长度
    """
    seq_lengths = torch.IntTensor([label_.size(0) for _, label_ in batch])
    seq_tensor = torch.cat([label_ for _, label_ in batch])
    img_tensor = torch.tensor([img_.numpy() for img_, _ in batch])
    return img_tensor, seq_tensor, seq_lengths


class CaptchaLoader:
    """数据集加载器"""

    def __init__(self):
        batch_size = config.batch_size
        nw = config.nw  # 取数进程数

        # 数据转换
        trainset = CaptchaSet(os.path.join(config.dataset_path, 'train'), transform)
        self.trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True,
                                                       num_workers=nw, collate_fn=collate_fn)

        testset = CaptchaSet(os.path.join(config.dataset_path, 'test'), transform)
        self.testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True,
                                                      num_workers=nw, collate_fn=collate_fn)

