#! -*- coding:utf-8 -*-
# /usr/bin/python3

"""
@Desc:
@Author: zhayongchun
@Date: 2023/12/21
"""
import os
import random
import shutil
import json
import pathlib

import click
from glob import glob
from PySide6.QtWidgets import QApplication, QListWidget, QHBoxLayout, QWidget, QPushButton, QLabel, QVBoxLayout, \
    QDialog, QLineEdit, QMessageBox
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import QSize, Qt

root_dir = pathlib.Path(__file__).parent.parent.parent


def get_predict_label(img_path: str):
    return "nothing"
    # res = requests.post("http://localhost:8000/captcha/predict", files={"files": open(img_path, "rb")})
    # return res.json()["data"][0]["label"]


class TagDialog(QDialog):
    def __init__(self, img_path):
        super().__init__()

        # 布局
        self.setWindowTitle('Make Tag Window')
        self.setFixedSize(QSize(400, 300))

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
        channel = os.path.basename(img_path).split("-")[0]
        self.text = get_predict_label(img_path)
        lbox = QHBoxLayout()
        lbox.addWidget(QLabel("Input"))
        label = QLabel(channel)
        label.setStyleSheet(f"color: {channel}")
        label.setFont(QFont("Arial", 20))
        lbox.addWidget(label)
        lbox.addWidget(QLabel(f"label or accept the prediction"))
        pred = QLabel(self.text)
        pred.setFont(QFont("Arial", 16))
        lbox.addWidget(pred)
        layout.addLayout(lbox)

        self.tag_input = QLineEdit()
        layout.addWidget(self.tag_input)

        # 创建一个按钮用于提交标签
        hbox = QHBoxLayout()
        tag_button = QPushButton("tag it")
        tag_button.clicked.connect(self.accept_tag)
        auto_button = QPushButton("accept")
        auto_button.clicked.connect(self.auto_tag)
        skip_button = QPushButton("skip")
        skip_button.clicked.connect(self.skip_tag)
        stop_button = QPushButton("stop")
        stop_button.clicked.connect(self.reject_tag)
        hbox.addWidget(tag_button)
        hbox.addWidget(auto_button)
        hbox.addWidget(skip_button)
        hbox.addWidget(stop_button)
        layout.addLayout(hbox)

        self.setLayout(layout)

    def auto_tag(self):
        self.done(22)

    def reject_tag(self):
        self.reject()

    def skip_tag(self):
        self.done(11)

    def accept_tag(self):
        if self.tag_input.text() == "":
            QMessageBox.warning(self, "Warning", "Please input your tag or skip this image", QMessageBox.Yes)
            return
        self.accept()

    def get_tag(self):
        # 获取用户输入的标签
        return self.tag_input.text()

    def get_auto_tag(self):
        return self.text

    def keyPressEvent(self, event):
        # 检查是否按下了回车键
        if event.key() == Qt.Key_Return:
            # 如果按下了回车键，执行相应的操作
            self.accept_tag()


class TagWindow(QWidget):
    def __init__(self, dataset_dir, target_dir, test_ratio=0.4):
        super().__init__()

        # 布局
        self.setWindowTitle('Make Tag Window')
        self.setFixedSize(QSize(800, 600))

        self.dataset_dir = dataset_dir
        self.target_dir = target_dir
        self.test_ratio = test_ratio

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
    def tag_filename(filename, tag, idx, channel):
        suffix = filename.split(".")[-1]
        return f"{channel}-{tag}-{idx}.{suffix}"

    def save_image(self, filename, tag, index):
        channel = filename.split("-")[0]
        tag_filename = self.tag_filename(filename, tag, index, channel)
        target_file_path = self.target_dir / tag_filename
        shutil.move(self.filename_map[filename], target_file_path)
        json_file = "train.json" if random.random() > self.test_ratio else "test.json"
        json_file_path = self.target_dir.parent / json_file
        json_list = []
        if json_file_path.exists():
            with open(json_file_path, "r") as f:
                json_list = json.load(f)
        channel = filename.split("-")[0]
        json_list.append({"filename": str(target_file_path), channel: tag})
        with open(json_file_path, "w") as f:
            json.dump(json_list, f)
        return tag_filename

    def process_file(self):
        # 检查未处理列表是否为空
        index = len(glob(f"{self.target_dir}/*.png"))
        for i in range(index, index + len(self.filename_map)):
            # 从未处理列表中取出一个文件
            file_item = self.unprocessed_list.currentItem() or self.unprocessed_list.item(0)
            filename = file_item.text()
            self.unprocessed_list.setCurrentItem(file_item)

            # 处理文件
            dialog = TagDialog(self.filename_map[filename])
            result = dialog.exec()
            # 删除指定行的项
            row = self.unprocessed_list.row(file_item)
            self.unprocessed_list.takeItem(row)
            self.unprocessed_list.setCurrentItem(self.unprocessed_list.item(0))

            if result == QDialog.Accepted or result == 22:
                self.unprocessed_label.setText(f"未处理:{self.unprocessed_list.count()}")
                tag = dialog.get_tag() if result == QDialog.Accepted else dialog.get_auto_tag()
                assert tag != "", "tag can not be empty"
                tag_filename = self.save_image(filename, tag, i)
                self.processed_list.addItem(tag_filename)
                self.processed_label.setText(f"已处理:{self.processed_list.count()}")
            elif result == QDialog.Rejected:
                break
            elif result == 11:
                self.unprocessed_label.setText(f"未处理:{self.unprocessed_list.count()}")
                continue
            else:
                raise ValueError("unknown result")


@click.command()
@click.option("--dataset-dir", type=str, default="origin")
@click.option("--output-dir", type=str, default="labeled")
@click.option("--test-ratio", type=float, default=0.4)
def main(dataset_dir, output_dir, test_ratio):
    dataset_dir = root_dir / "dataset" / dataset_dir
    save_dir = root_dir / "dataset" / output_dir / "images"
    os.makedirs(save_dir, exist_ok=True)

    app = QApplication()
    win = TagWindow(dataset_dir, save_dir, test_ratio)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
