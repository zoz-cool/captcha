#! -*- coding: utf-8 -*-

from typing import List

import paddle
import numpy as np
from PIL import Image
from loguru import logger
from ppqi import InferenceModel

from decoder import Decoder
from dataset import CaptchaDataset

data_util = CaptchaDataset(auto_gen=True)

# 加载模型
logger.info("Load model from inference/model...")
model = InferenceModel(
    modelpath='inference/model',
    use_gpu=False,
    use_mkldnn=True
)
model.eval()
# 解码器
decoder = Decoder(data_util.vocabulary)


def predict(img_list: List[Image.Image], colors: List[str]):
    batch_img = paddle.to_tensor([data_util.process_img(img) for img in img_list], dtype=paddle.float32)
    batch_color = paddle.to_tensor([data_util.process_channel(color) for color in colors], dtype=paddle.float32)
    outputs = paddle.to_tensor(model(batch_img, batch_color))
    outputs = paddle.nn.functional.softmax(outputs, axis=-1)
    # 解码获取识别结果
    pred_list = []
    for output in outputs:
        pred = decoder.ctc_greedy_decoder(output)
        pred_list.append(pred)
    return pred_list
