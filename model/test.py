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
