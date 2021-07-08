"""函数入口"""
import os
import torch
from PIL import Image
import argparse
from visdom import server

from .train import train
from .test import test
from .model import CTCModel
from .config import Config
from .dataset import CaptchaLoader, transform
from .utils import decode_output, calculate_acc

config = Config()


class Model:
    """模型入口"""

    @staticmethod
    def fit():
        train()

    @staticmethod
    def predict(x):
        """预测结果"""
        model = CTCModel(config.num_classes + 1, config.device_predict)
        # print('Load model from: ', config.model_path)
        model.load_state_dict(torch.load(config.model_path, map_location=config.device_predict))
        model.eval()

        x = transform(x).unsqueeze(0)
        outputs, _ = model(x)
        preds = decode_output(outputs)
        return preds[0]

    @staticmethod
    def test():
        test()

    @staticmethod
    def view():
        """可视化显示样本"""
        model_path = config.model_path
        model = CTCModel(config.num_classes + 1, device=config.device_predict)
        print('Load model from: ', model_path)
        model.load_state_dict(torch.load(model_path, map_location=config.device_predict))

        model.eval()
        test_loader = CaptchaLoader().testloader
        images, labels, label_length = next(iter(test_loader))
        with torch.no_grad():
            outputs, _ = model(images)
            preds, truths = calculate_acc(outputs, labels, label_length, call_label=True)

        import numpy as np
        import matplotlib.pyplot as plt
        plt.figure()
        for i, image in enumerate(images.numpy()[:16, :, :]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(np.array(image.transpose(1, 2, 0) * 255, dtype=np.uint8))
            plt.title(f'pred:{preds[i]},label:{truths[i]}')
            plt.axis('off')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input the necessary args.')
    parser.add_argument('-f', '--fit', help='fit mode', action='store_true')
    parser.add_argument('--train', help='train mode', action='store_true')
    parser.add_argument('-t', '--test', help='test mode', action='store_true')
    parser.add_argument('-p', '--predict', help='predict mode', type=str)
    parser.add_argument('-s', '--server', help='launch train monitor server', action='store_true')
    args = parser.parse_args()
    m = Model()
    if args.server:
        server.download_scripts_and_run()
    elif args.fit or args.train:
        m.fit()
    elif args.test:
        m.test()
        m.view()
    elif args.predict:
        path = args.predict
        img = Image.open(path)
        label = m.predict(img)
        print('标签：', os.path.basename(path).split('_')[0], '预测：', label)
    else:
        print(f'Usage: python main.py\n'
              f'\t[-f,--fit,--train] fit/train mode\n'
              f'\t[-t, --test] test mode\n'
              f'\t[-p, --predict path] predict mode\n'
              f'\t[-s --server] launch train monitor server')
