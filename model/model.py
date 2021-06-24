"""
端到端不定长验证码识别模型：残差网络提取特征，循环神经网络处理不定长标签对齐
"""

import torch
import torch.nn as nn
from .config import config


def _make_convolutional(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """构造卷积"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), (stride, stride), padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True))


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, (1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, (3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """特征提取网络"""

    def __init__(self):
        super().__init__()
        self.conv1 = _make_convolutional(config.sample_size[0], 32, 3, 1, 1)
        self.conv2 = _make_convolutional(32, 64, 3, 2, 1)
        self.layer1 = self._make_layer(64, 1)
        self.conv3 = _make_convolutional(64, 128, 3, 2, 1)
        self.layer2 = self._make_layer(128, 2)
        self.adapt_max_pool2d = nn.AdaptiveMaxPool2d((1, 15))  # 最大输出长度为C个字符，序列长度不小于：2C+1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    @staticmethod
    def _make_layer(in_channels, repeat_count):
        layers = []
        for _ in range(repeat_count):
            layers.append(ResidualBlock(in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.layer1(out)
        out = self.conv3(out)
        out = self.layer2(out)
        out = self.adapt_max_pool2d(out)
        return out


class CTCModel(nn.Module):
    """主模型"""

    def __init__(self, output_size, device=config.device_train):
        super().__init__()
        self.device = device
        self.feature_extractor = ResNet()
        self.num_layers = 2
        self.hidden_size = 80
        self.gru = nn.GRU(input_size=128, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * 2, output_size)
        self.log_softmax = nn.LogSoftmax(2)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.squeeze(2)
        out = out.permute(0, 2, 1)
        hidden = torch.zeros((self.num_layers * 2, out.size(0), self.hidden_size), device=self.device)
        out, _ = self.gru(out, hidden)
        out = self.fc(out)
        out = out.permute(1, 0, 2)
        output = self.log_softmax(out)
        output_lengths = torch.full(size=(out.size(1),), fill_value=out.size(0), dtype=torch.long, device=self.device)
        return output, output_lengths
