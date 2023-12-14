"""
从网络下载数据集
"""
import os
import pathlib
import shutil
import tarfile

import requests

URL = "https://yooongchun-share.oss-cn-beijing.aliyuncs.com/dataset/captcha/captcha_dataset_v2.tgz"


def download(url, save_path, decompress=True):
    """
    Download data from url, then save it to save_path
    """
    print(f'download data from {url}')
    save_dir = pathlib.Path(save_path).parent
    if not save_dir.is_dir():
        os.makedirs(save_dir, exist_ok=True)
    with requests.get(url, stream=True) as r:
        with open(save_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    if decompress:
        print(f'decompress file {save_path}...')
        with tarfile.open(save_path, "r:gz") as tar:
            tar.extractall(save_dir)


if __name__ == '__main__':
    root_dir = pathlib.Path(__file__).parent.parent.parent
    download(URL, f'{root_dir}/dataset/download/captcha_dataset_v2.tgz')
