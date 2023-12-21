#! -*- coding:utf-8 -*-
# /usr/bin/python3

"""
@Desc:
@Author: zhayongchun
@Date: 2023/12/21
"""
import os
import shutil
import pathlib

import click
import requests
from glob import glob
from PySide6.QtWidgets import QApplication, QListWidget, QHBoxLayout, QWidget, QPushButton, QLabel, QVBoxLayout, \
    QDialog, QLineEdit, QMessageBox
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QSize

root_dir = pathlib.Path(__file__).parent.parent.parent


def get_predict_label(img_path: str):
    res = requests.post('http://localhost:8000/captcha/predict', files={"files": open(img_path, 'rb')})
    return res.json()["data"][0]["label"]


class TagDialog(QDialog):
    def __init__(self, img_path):
        super().__init__()

        # 创建一个布局
        layout = QVBoxLayout()

        # 创建一个标签用于显示图片
        scale = 2
        self.img_label = QLabel()
        self.img_label.setFixedSize(QSize(int(120 * scale), int(50 * scale)))
        self.img_label.setScaledContents(True)

        self.img_label.setPixmap(QPixmap(img_path))

        layout.addWidget(self.img_label)

        # 创建一个文本框用于输入标签
        text = get_predict_label(img_path)
        label = QLabel(f"Input your label or accept the prediction: {text}")
        layout.addWidget(label)

        self.tag_input = QLineEdit()
        layout.addWidget(self.tag_input)

        # 创建一个按钮用于提交标签
        hbox = QHBoxLayout()
        self.submit_button = QPushButton("tag it")
        self.submit_button.clicked.connect(self.accept_tag)
        skip_button = QPushButton("skip")
        skip_button.clicked.connect(self.done(11))
        stop_button = QPushButton("stop")
        stop_button.clicked.connect(self.reject())
        hbox.addWidget(self.submit_button)
        hbox.addWidget(skip_button)
        hbox.addWidget(stop_button)
        layout.addLayout(hbox)

        self.setLayout(layout)

    def accept_tag(self):
        if self.tag_input.text() == "":
            QMessageBox.warning(self, "Warning", "Please input your tag or skip this image", QMessageBox.Yes)
            return
        self.accept()

    def get_tag(self):
        # 获取用户输入的标签
        return self.tag_input.text()


class TagWindow(QWidget):
    def __init__(self, dataset_dir, target_dir):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.target_dir = target_dir

        self.file_paths = glob(f"{self.dataset_dir}/*.png")
        self.filename_map = {os.path.basename(file_path): file_path for file_path in self.file_paths}

        # 创建两个列表
        self.unprocessed_list = QListWidget()
        self.processed_list = QListWidget()
        # 将文件添加到未处理列表
        self.unprocessed_list.addItems(list(self.filename_map.keys()))

        # 创建两个标签
        self.unprocessed_label = QLabel(f"未处理:{self.unprocessed_list.count()}")
        self.processed_label = QLabel(f"已处理:{self.processed_list.count()}")

        # 创建一个按钮
        self.process_button = QPushButton("Start")
        self.process_button.clicked.connect(self.process_file)

        quit_button = QPushButton("Exit")
        quit_button.clicked.connect(QApplication.instance().quit)

        # 创建一个布局并添加列表和按钮
        layout = QHBoxLayout()
        v1 = QVBoxLayout()
        v1.addWidget(self.unprocessed_label)
        v1.addWidget(self.unprocessed_list)
        layout.addLayout(v1)

        v2 = QVBoxLayout()
        v2.addWidget(self.process_button)
        v2.addWidget(quit_button)
        layout.addLayout(v2)

        v3 = QVBoxLayout()
        v3.addWidget(self.processed_label)
        v3.addWidget(self.processed_list)
        layout.addLayout(v3)

        self.setLayout(layout)

    @staticmethod
    def tag_filename(filename, tag, idx):
        channel = filename.split('-')[0]
        suffix = filename.split('.')[-1]
        return f"{channel}-{tag}-{idx}.{suffix}"

    def process_file(self):
        # 检查未处理列表是否为空
        index = len(glob(f"{self.target_dir}/*.png"))
        while self.unprocessed_list.count() > 0:
            # 从未处理列表中取出一个文件
            file_item = self.unprocessed_list.currentItem() or self.unprocessed_list.item(0)
            filename = file_item.text()
            self.unprocessed_label.setText(f"未处理:{self.unprocessed_list.count()}")

            # 处理文件
            dialog = TagDialog(self.filename_map[filename])
            result = dialog.exec()
            if result == QDialog.Accepted:
                tag = dialog.get_tag()
                assert tag != "", "tag can not be empty"

                tag_filename = self.tag_filename(filename, tag, index)
                index += 1
                target_file_path = self.target_dir / tag_filename
                shutil.copyfile(self.filename_map[filename], target_file_path)
                self.processed_label.setText(f"已处理:{self.processed_list.count()}")
                self.processed_list.addItem(tag_filename)
            elif result == QDialog.Rejected:
                break
            elif result == 11:
                continue


@click.command()
@click.option('--dataset-dir', type=str, default='origin')
@click.option('--output-dir', type=str, default='labeled')
def main(dataset_dir, output_dir):
    dataset_dir = root_dir / 'dataset' / dataset_dir
    save_dir = root_dir / 'dataset' / output_dir
    os.makedirs(save_dir, exist_ok=True)

    app = QApplication()
    win = TagWindow(dataset_dir, save_dir)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
