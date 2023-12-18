#! -*- coding: utf-8 -*-

import paddle
import paddle.nn as nn
from paddle.vision import models


class CustomFeatureNet(nn.Layer):
    """自定义的特征提取网络
    输入：[N, 3, 50, 120]
    输出：[N, 256, 29, 99]
    """

    def __init__(self):
        super(CustomFeatureNet, self).__init__()
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

    def forward(self, inputs):
        x = self.relu1(self.bn1(self.conv1(inputs)))
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
        return x


class Model(nn.Layer):
    def __init__(self, num_classes, max_len=6, feature_net="custom"):
        super(Model, self).__init__()
        feature_size = {"custom": [2872, 256],
                        "resnet18": [9, 512], "resnet34": [9, 512],
                        "resnet50": [9, 2048], "resnet101": [9, 2048], "resnet152": [9, 2048],
                        "resnet50_32x4d": [9, 2048], "resnet50_64x4d": [9, 2048],
                        "resnet101_32x4d": [9, 2048], "resnet101_64x4d": [9, 2048],
                        "resnet152_32x4d": [9, 2048], "resnet152_64x4d": [9, 2048],
                        "wide_resnet50_2": [9, 2048], "wide_resnet101_2": [9, 2048],
                        "mobilenet_v1": [9, 1024], "mobilenet_v2": [9, 1280],
                        "mobilenet_v3_small": [9, 576], "mobilenet_v3_large": [9, 960],
                        "shufflenet_v2_swish": [9, 1024], "shufflenet_v2_x0_25": [9, 512],
                        "shufflenet_v2_x0_33": [9, 512], "shufflenet_v2_x0_5": [9, 1024],
                        "shufflenet_v2_x1_0": [9, 1024], "shufflenet_v2_x1_5": [9, 1024],
                        "shufflenet_v2_x2_0": [9, 2048],
                        "vgg11": [4, 512], "vgg13": [4, 512], "vgg16": [4, 512], "vgg19": [4, 512]}
        assert feature_net in feature_size, f"feature_net must be one of {list(feature_size.keys())}"
        if feature_net == "custom":
            self.feature_net = CustomFeatureNet()
        else:
            net = eval(f"models.{feature_net}")
            self.feature_net = net(num_classes=-1, with_pool=False, pretrained=True)
        in_features, gru_input_size = feature_size[feature_net]
        self.fc = nn.Linear(in_features=in_features, out_features=max_len)
        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=128, num_layers=2, direction="bidirectional")
        self.output = nn.Linear(in_features=256, out_features=num_classes + 1)

    def forward(self, inputs, color_idx):
        x = self.feature_net(inputs)
        x = paddle.reshape(x, shape=(x.shape[0], x.shape[1], -1))
        # 将颜色类型向量和卷积层的输出拼接在一起
        color_channel = color_idx[:, None, None].astype("float32") * paddle.ones(shape=[x.shape[0], x.shape[1], 1])
        x = paddle.concat([x, color_channel], axis=-1)
        x = self.fc(x)
        x = paddle.transpose(x, perm=[0, 2, 1])
        y, h = self.gru(x)
        x = self.output(y)
        return x
