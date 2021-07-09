import sys
import os
import torch
from PIL import Image
from .model import CTCModel
from .config import Config
from .dataset import transform
from .utils import decode_output


config = Config()


def predict(x):
    """预测结果"""
    model = CTCModel(config.num_classes + 1, config.device_predict)
    model.load_state_dict(torch.load(config.model_path, map_location=config.device_predict))
    model.eval()

    x = transform(x).unsqueeze(0)
    outputs, _ = model(x)
    preds = decode_output(outputs)
    return preds[0]


if __name__ == '__main__':
    path = sys.argv[1]
    img = Image.open(path)
    label = predict(img)
    print('标签：', os.path.basename(path).split('_')[0], '预测：', label)
