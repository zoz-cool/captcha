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
import requests
from glob import glob
from PySide6.QtWidgets import QApplication, QListWidget, QHBoxLayout, QWidget, QPushButton, QLabel, QVBoxLayout, \
    QDialog, QLineEdit, QMessageBox, QProgressBar
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import QSize, Qt

root_dir = pathlib.Path(__file__).parent.parent.parent


def get_predict_label(img_path: str):
    res = requests.post("http://localhost:8000/captcha/predict", files={"files": open(img_path, "rb")})
    return res.json()["data"][0]["label"]


class TagDialog(QDialog):
    def __init__(self, img_path, enable_pred=False):
        super().__init__()

        self.enable_pred = enable_pred
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
        self.text = get_predict_label(img_path) if self.enable_pred else "-"
        lbox = QHBoxLayout()
        lbox.addWidget(QLabel("Input"))
        label = QLabel(channel.upper())
        label.setStyleSheet(f"color: {channel}")
        label.setFont(QFont("Arial", 20))
        lbox.addWidget(label)
        lbox.addWidget(QLabel(f"label or accept the prediction"))
        pred = QLabel(self.text)
        pred.setStyleSheet(f"color: {channel}")
        pred.setFont(QFont("Arial", 16))
        lbox.addWidget(pred)
        layout.addLayout(lbox)

        self.tag_input = QLineEdit()
        self.tag_input.setFixedSize(200, 50)
        self.tag_input.setFont(QFont("Arial", 20))
        self.tag_input.setStyleSheet(f"color: {channel}")
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
    def __init__(self, dataset_dir: pathlib.Path, target_dir: pathlib.Path, test_ratio: float = 0.4,
                 enable_pred: bool = False):
        super().__init__()

        self.enable_pred = enable_pred
        self.dataset_dir = dataset_dir
        self.target_dir = target_dir
        self.test_ratio = test_ratio

        self.file_paths = glob(f"{self.dataset_dir}/*.png")
        random.shuffle(self.file_paths)
        self.filename_map = {os.path.basename(file_path): file_path for file_path in self.file_paths}

        processed_files = [os.path.basename(file) for file in glob(f"{self.target_dir}/*.png")]
        self.processed_num = len(processed_files)
        self.total_num = len(self.file_paths) + self.processed_num

        # 各颜色标签的数量
        self.yellow_cnt = sum(1 for key in processed_files if key.startswith("yellow"))
        self.blue_cnt = sum(1 for key in processed_files if key.startswith("blue"))
        self.black_cnt = sum(1 for key in processed_files if key.startswith("black"))
        self.red_cnt = sum(1 for key in processed_files if key.startswith("red"))

        # ========UI布局============
        self.setWindowTitle('Make Tag Window')
        self.setFixedSize(QSize(800, 600))

        # 创建两个列表
        self.unprocessed_list = QListWidget()
        self.processed_list = QListWidget()
        # 将文件添加到未处理列表
        self.unprocessed_list.addItems(list(self.filename_map.keys()))

        # 创建两个标签
        self.unprocessed_label = QLabel(f"未处理:{self.unprocessed_list.count()}")
        self.processed_label = QLabel(f"已处理:{self.processed_list.count()}（总计{self.processed_num}）")

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

        v2.addStretch(1)
        v2.addWidget(self.process_button)
        v2.addWidget(quit_button)
        v2.addStretch(1)
        layout.addLayout(v2)

        v3 = QVBoxLayout()
        v3.addWidget(self.processed_label)
        v3.addWidget(self.processed_list)
        layout.addLayout(v3)

        # 创建一个进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(self.processed_num)
        self.progress_bar.setMaximum(self.total_num)
        self.progress_bar.setFormat("Progress: %p%")

        # 标签数量重
        h2 = QHBoxLayout()
        self.yellow_lab = QLabel(f"YELLOW: {self.yellow_cnt}")
        self.yellow_lab.setStyleSheet("color: yellow")
        self.yellow_lab.setFont(QFont("Arial", 16))
        self.blue_lab = QLabel(f"BLUE: {self.blue_cnt}")
        self.blue_lab.setStyleSheet("color: blue")
        self.blue_lab.setFont(QFont("Arial", 16))
        self.black_lab = QLabel(f"BLACK: {self.black_cnt}")
        self.black_lab.setStyleSheet("color: black")
        self.black_lab.setFont(QFont("Arial", 16))
        self.red_lab = QLabel(f"RED: {self.red_cnt}")
        self.red_lab.setStyleSheet("color: red")
        self.red_lab.setFont(QFont("Arial", 16))

        h2.addStretch(3)
        h2.addWidget(self.yellow_lab)
        h2.addStretch(1)
        h2.addWidget(self.blue_lab)
        h2.addStretch(1)
        h2.addWidget(self.black_lab)
        h2.addStretch(1)
        h2.addWidget(self.red_lab)
        h2.addStretch(3)

        vbox = QVBoxLayout()
        vbox.addWidget(self.progress_bar)
        vbox.addLayout(h2)

        vbox.addLayout(layout)

        self.setLayout(vbox)

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
            with open(json_file_path, "r", encoding="utf-8") as f:
                json_list = json.load(f)
        channel = filename.split("-")[0]

        json_list.append({"path": self.target_dir.name + "/" + filename, channel: tag})
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(json_list, f, ensure_ascii=False, indent=4)
        return tag_filename

    def select_item_based_on_ratio(self):
        # 计算每种样本的比例
        total = self.yellow_cnt + self.blue_cnt + self.black_cnt + self.red_cnt
        channels = ["yellow", "blue", "black", "red"]
        ratios = [self.yellow_cnt / total, self.blue_cnt / total, self.black_cnt / total, self.red_cnt / total]
        acc_ratios = [ratios[i - 1] + ratios[i] for i in range(1, 4)]
        channel = channels[-1]
        for i in range(4):
            if acc_ratios[-1] * random.random() <= acc_ratios[i]:
                channel = channels[i]
                break
        assert channel is not None, "channel is None"
        # 根据比例随机选择元素
        yellow_items = [self.unprocessed_list.item(i) for i in range(self.unprocessed_list.count()) if
                        self.unprocessed_list.item(i).text().startswith('yellow')]
        blue_items = [self.unprocessed_list.item(i) for i in range(self.unprocessed_list.count()) if
                      self.unprocessed_list.item(i).text().startswith('blue')]
        black_items = [self.unprocessed_list.item(i) for i in range(self.unprocessed_list.count()) if
                       self.unprocessed_list.item(i).text().startswith('black')]
        red_items = [self.unprocessed_list.item(i) for i in range(self.unprocessed_list.count()) if
                     self.unprocessed_list.item(i).text().startswith('red')]
        if channel == "yellow" and yellow_items:
            selected_item = random.choice(yellow_items)
            self.yellow_cnt += 1
            return selected_item
        elif channel == "yellow":
            channel = "blue"
        if channel == "blue" and blue_items:
            selected_item = random.choice(blue_items)
            self.blue_cnt += 1
            return selected_item
        elif channel == "blue":
            channel = "black"
        if channel == "black" and black_items:
            selected_item = random.choice(black_items)
            self.black_cnt += 1
            return selected_item
        elif channel == "black":
            channel = "red"
        if channel == "red" and red_items:
            selected_item = random.choice(red_items)
            self.red_cnt += 1
            return selected_item
        else:
            raise ValueError(
                f"unknown channel: {channel}, {len(yellow_items)}, {len(blue_items)}, {len(black_items)}, {len(red_items)}")

    def process_file(self):
        # 检查未处理列表是否为空

        for i in range(self.processed_num, self.total_num):
            # 从未处理列表中取出一个文件
            file_item = self.select_item_based_on_ratio()
            filename = file_item.text()
            self.unprocessed_list.setCurrentItem(file_item)

            # 处理文件
            dialog = TagDialog(self.filename_map[filename], self.enable_pred)
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
                self.processed_label.setText(
                    f"已处理:{self.processed_list.count()}（总计{self.processed_list.count() + self.processed_num}）")
                self.progress_bar.setValue(i)
                self.yellow_lab.setText(f"YELLOW: {self.yellow_cnt}")
                self.blue_lab.setText(f"BLUE: {self.blue_cnt}")
                self.black_lab.setText(f"BLACK: {self.black_cnt}")
                self.red_lab.setText(f"RED: {self.red_cnt}")
            elif result == QDialog.Rejected:
                break
            elif result == 11:
                self.unprocessed_label.setText(f"未处理:{self.unprocessed_list.count()}")
                self.progress_bar.setValue(i)
                continue
            else:
                raise ValueError("unknown result")


dark_stylesheet = """
    QWidget {
        background-color: #2b2b2b;
        color: #b1b1b1;
    }
    QLineEdit {
        background-color: #353535;
        color: #b1b1b1;
    }
    QPushButton {
        background-color: #353535;
        color: #b1b1b1;
    }
    QListWidget {
        background-color: #353535;
        color: #b1b1b1;
    }
    QProgressBar {
        background-color: #353535;
        color: #b1b1b1;
    }
"""


@click.command()
@click.option("--dataset-dir", type=str, default="origin")
@click.option("--output-dir", type=str, default="labeled")
@click.option("--test-ratio", type=float, default=0.4)
@click.option("--enable-pred", type=bool, default=False)
def main(dataset_dir, output_dir, test_ratio, enable_pred):
    dataset_dir = root_dir / "dataset" / dataset_dir
    save_dir = root_dir / "dataset" / output_dir / "images"
    os.makedirs(save_dir, exist_ok=True)

    app = QApplication()
    app.setStyleSheet(dark_stylesheet)

    win = TagWindow(dataset_dir, save_dir, test_ratio, enable_pred)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
