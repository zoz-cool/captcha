"""
分割颜色通道
"""
import io
import os
import pathlib
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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


def test_split_channel(img_path):
    img = Image.open(img_path)
    channel_data = split_channel(img)
    plt.figure()
    grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.3)
    plt.subplot(grid[0, :])
    plt.imshow(np.array(img, dtype=np.uint8))
    plt.xlabel('Origin image')
    plt.xticks([])
    plt.yticks([])
    for idx, (color, img) in enumerate(channel_data.items()):
        idx += 1
        plt.subplot(grid[idx//2+idx % 2, idx % 2-1])
        plt.imshow(np.array(img, dtype=np.uint8))
        plt.xlabel(f'channel:{color}')
        plt.xticks([])
        plt.yticks([])

    plt.show()


def run_all(origin_dir, save_dir, filter_size=300):
    """Split all images to r/g/b channel"""
    assert os.path.isdir(origin_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    for i, filepath in enumerate(glob(f"{origin_dir}/*.png")):
        print(f'[{i+1}] Processing on {os.path.basename(filepath)}')
        img = Image.open(filepath)
        res_dict = split_channel(img)
        for color, img in res_dict.items():
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            size = sys.getsizeof(img_bytes.getvalue())
            if size > filter_size:
                img.save(f'{save_dir}/{pathlib.Path(filepath).stem}-{color}.png')


if __name__ == '__main__':
    # test_split_channel('dataset/origin/1676449863.623534.png')
    dataset_dir = pathlib.Path(__file__).parent.parent / 'dataset'
    run_all(dataset_dir / 'origin', dataset_dir / 'splitted')
