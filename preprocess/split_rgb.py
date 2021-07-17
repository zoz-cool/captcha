"""
分割颜色通道
"""
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def split_channel(img, upper=220, lower=60):
    """按颜色分离"""
    img_red = np.array(img.copy())
    img_red[~((img_red[:, :, 0] > upper) & (img_red[:, :, 1] < lower) & (img_red[:, :, 2] < lower))] = 255
    img_blue = np.array(img.copy())
    img_blue[~((img_blue[:, :, 0] < lower) & (img_blue[:, :, 1] < lower) & (img_blue[:, :, 2] > upper))] = 255
    img_black = np.array(img.copy())
    img_black[~((img_black[:, :, 0] < lower) & (img_black[:, :, 1] < lower) & (img_black[:, :, 2] < lower))] = 255

    img_yellow = np.array(img.copy())
    img_yellow[~((img_yellow[:, :, 0] > upper) & (img_yellow[:, :, 1] > upper) & (img_yellow[:, :, 2] < lower))] = 255
    
    img_red = Image.fromarray(img_red)
    img_yellow = Image.fromarray(img_yellow)
    img_blue = Image.fromarray(img_blue)
    img_black = Image.fromarray(img_black)

    return {'red': img_red, 'yellow': img_yellow, 'blue': img_blue, 'black': img_black}


def test_split_channel():
    from random import choice
    origin_dir = 'dataset/origin/'
    path = os.path.join(origin_dir, choice(os.listdir(origin_dir)))
    img = Image.open(path)
    channel_data = split_channel(img)
    plt.figure()
    grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.3)
    plt.subplot(grid[0, :])
    plt.imshow(img)
    plt.xlabel('Origin image')
    plt.xticks([])
    plt.yticks([])
    for idx, (color, img) in enumerate(channel_data.items()):
        idx += 1
        plt.subplot(grid[idx//2+idx % 2, idx % 2-1])
        plt.imshow(img)
        plt.xlabel(f'channel:{color}')
        plt.xticks([])
        plt.yticks([])

    plt.show()


if __name__ == '__main__':
    test_split_channel()
