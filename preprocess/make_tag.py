"""
给图片打标签
"""
import os
import pathlib
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from queue import Queue

import click
import requests
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QApplication, QInputDialog, QLabel, QLineEdit,
                               QMainWindow, QPushButton, QVBoxLayout, QWidget)

root_dir = pathlib.Path(__file__).parent.parent
cached_label = Queue()


def get_pred_label(img_path):
    res = requests.post('https://www.yooongchun.com/api/captcha', files={"file": open(img_path, 'rb')})
    res_dict = res.json()
    return img_path, res_dict['predict_label'], round(res_dict['confidence'], 3)


def make_cache_label(files, worker_num=50):
    """pre query image label"""
    print('start to cache label...')
    with ThreadPoolExecutor(max_workers=worker_num) as pool:
        tasks = [pool.submit(get_pred_label, filepath) for filepath in files]
        for task in as_completed(tasks):
            img_path, label, ci = task.result()
            cached_label.put((img_path, label, ci))
    print(f'cache label done, total cached file {len(files)}!')


class TagWindow(QMainWindow):
    def __init__(self, files_dir, target_dir):
        super().__init__()
        self.files = glob(f"{files_dir}/*.png")
        self.target_dir = pathlib.Path(target_dir)
        self.exist_count = len(glob(f"{target_dir}/*.png"))
        self.img_iter = self.get_img_iter()

        # 布局
        self.setWindowTitle('Make Tag Window')
        self.setFixedSize(QSize(800, 600))

        self.label = QLabel('IMAGE HERE')
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scale = 2
        self.label.setFixedSize(QSize(int(120*scale), int(50*scale)))
        self.label.setScaledContents(True)
        btn = QPushButton('START')
        btn.clicked.connect(self.next_img)
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(btn)
        widget = QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)

    def get_img_iter(self):
        for i, filepath in enumerate(self.files):
            _, label, ci = get_pred_label(filepath)
            yield i+1, label, ci, filepath

    def next_img(self):
        """recur image path"""
        try:
            idx, auto_label, ci, file_path = next(self.img_iter)
            self.label.setPixmap(QPixmap(file_path))
            title = f"[{idx}/{len(self.files)}]"
            tip = f"请输入图片中的字符:(ci={ci})"
            checked_label, ok = QInputDialog.getText(self, title, tip, QLineEdit.Normal, auto_label)
            if ok and checked_label: # click OK
                save_path = self.target_dir / f"{checked_label}_{self.exist_count+idx}.png"
                shutil.move(file_path, save_path)
                print('tag file', str(file_path))
            elif not ok:
                print('remove file', str(file_path))
                os.remove(file_path)
            elif checked_label == "":
                print("stop iter!")
                self.close()
                return
            self.next_img()
        except StopIteration:
            return


def manual_tag(files_dir, target_dir):
    """make tag manually"""
    app = QApplication()
    win = TagWindow(files_dir, target_dir)
    win.show()
    app.exec_()


def auto_tag(files_dir, target_dir, hard_dir, min_ci=0.90, test_ratio=0.3):
    """make tag automatically"""
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(hard_dir, exist_ok=True)
    ori_files = glob(f"{files_dir}/*.png")
    ori_count = len(ori_files)
    count = len(glob(f"{target_dir}/*.png"))
    hard_count = len(glob(f"{hard_dir}/*.png"))
    # pre query image label
    threading.Thread(target=make_cache_label, args=(ori_files,)).start()
    process_count = 0
    while process_count < ori_count:
        while not cached_label.empty():
            process_count += 1
            img_path, label, ci = cached_label.get()
            taged = ['-']
            if ci > min_ci:
                save_path = pathlib.Path(target_dir, f'{label}_auto_ci{ci}_{count+process_count}.png')
                taged = '[TAGED]'
            else:
                save_path = pathlib.Path(hard_dir, f'{label}_auto_ci{ci}_{hard_count+process_count}.png')
            shutil.move(img_path, save_path)
            print(f'[{process_count}/{ori_count}]Automatically make tag to {os.path.basename(img_path)} {taged}')
        print('wait cache label...')
        time.sleep(1)


def generate_label_file(data_dir, save_label_path):
    """Generate label file for ALL image in data dir"""
    with open(save_label_path, 'w') as fp:
        for file_path in glob(f"{data_dir}/*.png", recursive=True):
            path = pathlib.Path(file_path)
            label = file_path.split('/')[-1].split('_')[0]
            fp.write(f'{path.parent.name}/{path.name}\t{label}\n')


@click.command()
@click.option('--mode', required=True, type=click.Choice(['auto-tag', 'man-tag', 'gen-label']))
@click.option('--data-dir', type=str)
def run(mode, data_dir):
    if mode == 'auto-tag':
        splitted_dir = root_dir / 'dataset/splitted'
        target_dir = root_dir / 'dataset/auto_labeled'
        hard_dir = root_dir / 'dataset/hard'
        auto_tag(splitted_dir, target_dir, hard_dir)
    elif mode == 'man-tag':
        splitted_dir = root_dir / 'dataset/hard'
        target_dir = root_dir / 'dataset/train'
        manual_tag(splitted_dir, target_dir)
    elif mode == 'gen-label':
        if data_dir is None:
            raise RuntimeError('Must specify dataset dir')
        else:
            prefix = pathlib.Path(data_dir).parent
            print(f'Generate label file in {prefix}/gt_label.txt')
            generate_label_file(data_dir, f'{prefix}/gt_label.txt')


if __name__ == '__main__':
    run()
