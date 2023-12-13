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

    def forward(self, inputs_with_color_idx):
        x = inputs_with_color_idx[:3, :, :]
        color_idx = inputs_with_color_idx[3, 0, 0]
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
        # 将颜色类型向量和卷积层的输出拼接在一起
        color_channel = color_idx * paddle.ones(shape=[x.shape[0], x.shapep[1], 1], dtype="float32")
        x = paddle.concat([x, color_channel], axis=-1)
        x = self.fc(x)
        x = paddle.transpose(x, perm=[0, 2, 1])
        y, h = self.gru(x)
        x = self.output(y)
        x = x.transpose(perm=[1, 0, 2])
        return x
