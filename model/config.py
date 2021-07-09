"""
在此配置模型参数
"""

import os
import torch


class Config(object):
    """模型参数"""
    # 类别：[2,3,4,5,6,7,8,9,A,B,C,D,E,F,G,H,I,J,K,L,M,N,P,Q,R,S,T,U,V,W,X,Y,Z]
    # 33个字符（去掉易混淆的0,1,O）
    charset = ['2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    classes = {c: i for i, c in enumerate(charset)}  # 通过字符查找索引
    classes.update({i: c for i, c in enumerate(charset)})  # 通过索引查找字符
    num_classes = len(charset)
    batch_size = 128
    sample_size = (1, 50, 120)  # 样本大小
    num_epochs = 100
    nw = max(os.cpu_count(), 8)
    learning_rate = 0.01
    model_path = os.path.join(os.path.dirname(__file__), 'model/checkpoint/captcha.ctc-model-2.pth')  # 训练参数保存路径
    dataset_path = 'dataset/captcha/'  # 数据集路径
    # 选择设备：GPU或CPU
    device_train = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_test = device_train
    device_predict = torch.device('cpu')
    # 每5个epoch保存一次训练结果
    save_n_epoch = 5

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance
