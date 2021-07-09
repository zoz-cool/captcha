"""
训练模型
"""

import os
import torch
import visdom
from tqdm import tqdm
import numpy as np
from collections import deque
from torchsummary import summary

from .test import test
from .config import Config
from .model import CTCModel
from .utils import calculate_acc
from .dataset import CaptchaLoader


def train():
    """训练模型"""
    config = Config()
    model = CTCModel(config.num_classes + 1).to(config.device_train)
    summary(model, config.sample_size)
    criterion = torch.nn.CTCLoss(blank=config.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    print(f'Training on device: {config.device_train}')
    model_path = config.model_path
    if os.path.isfile(model_path):
        print('Load model from: ', model_path)
        model.load_state_dict(torch.load(model_path))
        model.train()

    vis = visdom.Visdom(env='captcha_model')
    train_loader = CaptchaLoader().trainloader
    global_steps = 0
    for epoch in range(config.num_epochs):
        print(f'Epoch:{epoch + 1}/{config.num_epochs}')
        pbar = tqdm(train_loader)
        running_loss = deque(maxlen=10)
        for i, (images, labels, label_length) in enumerate(pbar):
            images = images.to(config.device_train)
            labels = labels.to(config.device_train)
            label_length = label_length.to(config.device_train)
            outputs, output_length = model(images)
            loss = criterion(outputs, labels, output_length, label_length)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)
            running_loss.append(loss.item())
            corr, tot = calculate_acc(outputs, labels, label_length)

            global_steps += 1
            if len(running_loss) == 10:
                vis.line([np.nanmean(running_loss)], [global_steps], win='train_loss', update='append',
                         opts={'title': 'train loss', 'xlabel': 'steps', 'ylabel': 'value'})
                vis.line([corr / tot], [global_steps], win='train_acc', update='append',
                         opts={'title': 'train accuracy', 'xlabel': 'steps', 'ylabel': 'pct'})
            msg = f'loss:{np.around(np.nanmean(running_loss), 5)} acc:{np.around(corr / tot * 100, 2)}%'
            vis.text(text=msg, win='train_log', append=True, opts={'title': 'train log'})
            pbar.set_description_str(msg)
        # calculate test acc each epoch
        test_acc = test(model)
        model.train()
        vis.line([test_acc], [global_steps], win='test_acc', update='append',
                 opts={'title': 'test accuracy', 'xlabel': 'steps', 'ylabel': 'pct'})

        # save model
        if (epoch + 1) % config.save_n_epoch == 0:
            print(f'Save model to: {model_path}')
            if not os.path.isdir(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    train()
