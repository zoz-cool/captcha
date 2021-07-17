"""
测试样本正确率
"""
import torch
from .config import Config
from .model import CTCModel
from .dataset import CaptchaLoader
from tqdm import tqdm
from .utils import calculate_acc

config = Config()


def test(model=None):
    """测试模型"""
    device = config.device_test
    if model is None:
        model_path = config.model_path
        model = CTCModel(config.num_classes + 1, device=device).to(device)

        print('Load model from: ', model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    test_loader = CaptchaLoader().testloader
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels, label_length in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            corr, tot = calculate_acc(outputs, labels, label_length)
            correct += corr
            total += tot
        print(f'Test accuracy on {total} examples:{round(correct / total * 100, 2)}%')
        return correct / total


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
    test()
    view()
