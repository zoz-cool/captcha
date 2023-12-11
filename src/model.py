#! -*- coding: utf-8 -*-

import paddle
import paddle.nn as nn


class Model(nn.Layer):
    def __init__(self, num_classes, max_len=6):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2D(32)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=1)

        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2D(64)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=1)

        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2D(128)
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=1)

        self.conv4 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2D(256)
        self.pool4 = nn.MaxPool2D(kernel_size=2, stride=1)

        self.conv5 = nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2D(256)
        self.pool5 = nn.MaxPool2D(kernel_size=2, stride=1)

        self.conv6 = nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3)
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2D(256)
        self.pool6 = nn.MaxPool2D(kernel_size=2, stride=1)

        self.conv7 = nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3)
        self.relu7 = nn.ReLU()
        self.bn7 = nn.BatchNorm2D(256)
        self.pool7 = nn.MaxPool2D(kernel_size=2, stride=1)

        self.fc = nn.Linear(in_features=2871, out_features=max_len * 2 + 1)

        self.gru = nn.GRU(input_size=256, hidden_size=128)

        self.output = nn.Linear(in_features=128, out_features=num_classes + 1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.pool6(x)
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.pool7(x)
        x = paddle.reshape(x, shape=(x.shape[0], x.shape[1], -1))
        x = self.fc(x)
        x = paddle.transpose(x, perm=[0, 2, 1])
        y, h = self.gru(x)
        x = self.output(y)
        x = x.transpose(perm=[1, 0, 2])
        return x


def _make_convolutional(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """构造卷积"""
    return nn.Sequential(
        nn.Conv2D(in_channels, out_channels, (kernel_size, kernel_size), (stride, stride), padding),
        nn.BatchNorm2D(out_channels), nn.LeakyReLU())


class ResidualBlock(nn.Layer):
    """残差块"""

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, in_channels // 2, (1, 1), stride=(1, 1), padding=0)
        self.bn1 = nn.BatchNorm2D(in_channels // 2)
        self.conv2 = nn.Conv2D(in_channels // 2, in_channels, (3, 3), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2D(in_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        return out


class ResNet(nn.Layer):
    """特征提取网络"""

    def __init__(self, max_len=6):
        super().__init__()
        self.conv1 = _make_convolutional(3, 32, 3, 1, 1)
        self.conv2 = _make_convolutional(32, 64, 3, 2, 1)
        self.layer1 = self._make_layer(64, 1)
        self.conv3 = _make_convolutional(64, 128, 3, 2, 1)
        self.layer2 = self._make_layer(128, 2)
        self.adapt_max_pool2d = nn.AdaptiveMaxPool2D((1, 2 * max_len + 1))  # 最大输出长度为C个字符，序列长度不小于：2C+1
        for m in self.sublayers(include_self=True):
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal(m.weight, nonlinearity='leaky_relu')

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


class Model2(nn.Layer):
    """主模型"""

    def __init__(self, num_classes, max_len=6):
        super().__init__()
        self.resnet = ResNet(max_len=max_len)
        self.num_layers = 2
        self.hidden_size = 80
        self.gru = nn.GRU(input_size=128, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          direction="bidirect")
        self.fc = nn.Linear(self.hidden_size * 2, num_classes + 1)
        self.log_softmax = nn.LogSoftmax(2)

    def forward(self, x):
        out = self.resnet(x)
        out = out.squeeze(2)
        out = out.transpose(perm=[0, 2, 1])
        hidden = paddle.zeros((self.num_layers * 2, out.shape[0], self.hidden_size))
        out, _ = self.gru(out, hidden)
        out = self.fc(out)
        out = out.transpose(perm=[1, 0, 2])
        # out = self.log_softmax(out)
        return out
