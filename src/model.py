"""
Description: 可变长验证码识别模型CRNN+CTC
Author: yczha
Email: yooongchun@foxmail.com
"""

import paddle
import paddle.nn as nn
from paddle.vision.models import resnet34


class CaptchaModel(nn.Layer):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, inputs):
        pass
