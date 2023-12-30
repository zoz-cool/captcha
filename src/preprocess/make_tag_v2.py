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
from loguru import logger
from PySide6.QtWidgets import QApplication, QListWidget, QHBoxLayout, QWidget, QPushButton, QLabel, QVBoxLayout, \
    QDialog, QLineEdit, QMessageBox, QProgressBar
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import QSize, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

root_dir = pathlib.Path(__file__).parent.parent.parent


def get_predict_label(img_path: str):
    res = requests.post("http://localhost:8000/captcha/predict", files={"files": open(img_path, "rb")})
    return res.json()["data"][0]["label"]


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor=None)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class TagAllDialog(QDialog):
    """one picture with multi tag"""

    def __init__(self, img_path, enable_pred=False):
        super().__init__()

        self.img_path = img_path
        self.enable_pred = enable_pred
        # Layout
        self.setWindowTitle('Make Multi-Tag Window')
        self.setFixedSize(QSize(400, 400))

        # make a new layout
        layout = QVBoxLayout()

        # make a label to show the image
        scale = 2
        self.img_label = QLabel()
        self.img_label.setFixedSize(QSize(int(120 * scale), int(50 * scale)))
        self.img_label.setScaledContents(True)

        self.img_label.setPixmap(QPixmap(img_path))

        layout.addWidget(self.img_label)
        layout.addStretch(1)
        # add all channel color input line
        self.inputs = {}
        for color in ["black", "red", "blue", "yellow"]:
            hbox = QHBoxLayout()
            label = QLabel(color.upper().ljust(15))
            # label.setStyleSheet(f"color: {color}")
            label.setFont(QFont("Arial", 16))
            hbox.addWidget(label)
            text_input = MyLineEdit()
            text_input.setFixedSize(120, 30)
            text_input.setFont(QFont("Arial", 16))
            text_input.setStyleSheet(f"color: {color}")
            self.inputs[color] = text_input
            hbox.addWidget(text_input)
            # hbox.addStretch(1)
            layout.addLayout(hbox)

        # layout.addStretch(1)
        # function button
        hbox2 = QHBoxLayout()
        tag_button = QPushButton("tag it")
        tag_button.clicked.connect(self.accept_tag)
        auto_button = QPushButton("accept")
        auto_button.clicked.connect(self.auto_tag)
        skip_button = QPushButton("skip")
        skip_button.clicked.connect(self.skip_tag)
        stop_button = QPushButton("stop")
        stop_button.clicked.connect(self.reject_tag)
        hbox2.addWidget(tag_button)
        hbox2.addWidget(auto_button)
        hbox2.addWidget(skip_button)
        hbox2.addWidget(stop_button)
        layout.addLayout(hbox2)

        self.setLayout(layout)

    def auto_tag(self):
        self.done(22)

    def reject_tag(self):
        self.reject()

    def skip_tag(self):
        self.done(11)

    def accept_tag(self):
        self.accept()

    def get_tag(self):
        # get user input tag
        return {color: self.inputs[color].text().upper() for color in self.inputs}

    def get_auto_tag(self):
        return {color: get_predict_label(self.img_path) for color in self.inputs}

    def keyPressEvent(self, event):
        # check if press enter
        if event.key() == Qt.Key_Return:
            # if press enter, do the corresponding operation
            self.accept_tag()


class MyLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super(MyLineEdit, self).__init__(*args, **kwargs)
        self.textChanged.connect(self.on_text_changed)

    def on_text_changed(self, text):
        self.setText(text.upper())


class TagDialog(QDialog):
    def __init__(self, img_path, enable_pred=False):
        super().__init__()

        self.enable_pred = enable_pred
        # Layout
        self.setWindowTitle('Make Tag Window')
        self.setFixedSize(QSize(400, 300))

        layout = QVBoxLayout()

        scale = 2
        self.img_label = QLabel()
        self.img_label.setFixedSize(QSize(int(120 * scale), int(50 * scale)))
        self.img_label.setScaledContents(True)

        self.img_label.setPixmap(QPixmap(img_path))

        layout.addWidget(self.img_label)

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

        self.tag_input = MyLineEdit()
        self.tag_input.setFixedSize(200, 50)
        self.tag_input.setFont(QFont("Arial", 20))
        self.tag_input.setStyleSheet(f"color: {channel}")
        layout.addWidget(self.tag_input)

        # function buttons
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
        return self.tag_input.text().upper()

    def get_auto_tag(self):
        return self.text

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.accept_tag()


class TagWindow(QWidget):
    def __init__(self, dataset_dir: pathlib.Path, target_dir: pathlib.Path, test_ratio: float = 0.4,
                 enable_pred: bool = False, multi_tag: bool = False):
        super().__init__()

        # 创建一个新的MplCanvas实例
        self.canvas = MplCanvas(self, width=3, height=2, dpi=100)
        self.canvas.axes.axis('off')  # close the axis

        self.enable_pred = enable_pred
        self.multi_tag = multi_tag
        self.dataset_dir = dataset_dir
        self.target_dir = target_dir
        self.test_ratio = test_ratio

        self.file_paths = glob(f"{self.dataset_dir}/*.png")
        random.shuffle(self.file_paths)
        self.filename_map = {os.path.basename(file_path): file_path for file_path in self.file_paths}

        processed_files = [os.path.basename(file) for file in glob(f"{self.target_dir}/*.png")]
        self.processed_num = len(processed_files)
        self.total_num = len(self.file_paths) + self.processed_num

        # count the number of each tag
        self.yellow_cnt = sum(1 for key in processed_files if key.startswith("yellow"))
        self.blue_cnt = sum(1 for key in processed_files if key.startswith("blue"))
        self.black_cnt = sum(1 for key in processed_files if key.startswith("black"))
        self.red_cnt = sum(1 for key in processed_files if key.startswith("red"))

        # ========UI Layout============
        self.setWindowTitle('Make Tag Window')
        self.setFixedSize(QSize(800, 600))

        # create two list
        self.unprocessed_list = QListWidget()
        self.processed_list = QListWidget()
        # add all file name to the list
        self.unprocessed_list.addItems(list(self.filename_map.keys()))

        self.unprocessed_label = QLabel(f"Unprocessed:{self.unprocessed_list.count()}")
        self.processed_label = QLabel(f"Processed:{self.processed_list.count()}(Total {self.processed_num})")

        # function button
        self.single_tag = QPushButton("Single Tag")
        self.single_tag.clicked.connect(self.process_single_tag)

        self.multi_tag_btn = QPushButton("Multi Tag")
        self.multi_tag_btn.clicked.connect(self.process_multi_tag)

        quit_button = QPushButton("Exit")
        quit_button.clicked.connect(QApplication.instance().quit)

        # layout
        layout = QHBoxLayout()
        v1 = QVBoxLayout()
        v1.addWidget(self.unprocessed_label)
        v1.addWidget(self.unprocessed_list)
        layout.addLayout(v1)

        v2 = QVBoxLayout()

        v2.addStretch(1)
        v2.addWidget(self.single_tag)
        v2.addWidget(self.multi_tag_btn)
        v2.addWidget(quit_button)
        v2.addStretch(1)
        layout.addLayout(v2)

        v3 = QVBoxLayout()
        v3.addWidget(self.processed_label)
        v3.addWidget(self.processed_list)
        layout.addLayout(v3)

        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(self.processed_num)
        self.progress_bar.setMaximum(self.total_num)
        self.progress_bar.setFormat("Progress: %p%")

        vbox = QVBoxLayout()
        vbox.addWidget(self.progress_bar)
        # 将这个canvas添加到你的布局中
        vbox.addWidget(self.canvas)

        vbox.addLayout(layout)

        self.setLayout(vbox)
        self.update_figure()

    @staticmethod
    def tag_filename(filename, tag, idx, channel):
        suffix = filename.split(".")[-1]
        return f"{channel}-{tag}-{idx}.{suffix}"

    def update_figure(self):
        # 在这个函数中，你可以更新你的图形

        sizes = [self.yellow_cnt, self.blue_cnt, self.black_cnt, self.red_cnt]
        labels = ['Yellow', 'Blue', 'Black', 'Red']

        self.canvas.axes.clear()
        wedges, texts, autotexts = self.canvas.axes.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        self.canvas.axes.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # add annotation
        # 添加每种颜色标签的具体数量
        legend_labels = [f'{label}: {size}' for label, size in zip(labels, sizes)]
        self.canvas.axes.legend(wedges, legend_labels, title="Colors", loc="center left",
                                bbox_to_anchor=(0.8, 0, 0.5, 1))

        self.canvas.draw()

    def save_image(self, filename, tag, index, channel=None):
        if channel is None:
            channel = filename.split("-")[0]
        tag_filename = self.tag_filename(filename, tag, index, channel)
        target_file_path = self.target_dir / tag_filename
        logger.info(f"move image from {os.path.basename(self.filename_map[filename])} to {target_file_path.name}")
        shutil.copy(self.filename_map[filename], target_file_path)
        json_file = "train.json" if random.random() > self.test_ratio else "test.json"
        json_file_path = self.target_dir.parent / json_file
        json_list = []
        if json_file_path.exists():
            with open(json_file_path, "r", encoding="utf-8") as f:
                json_list = json.load(f)
        record = {"path": self.target_dir.name + "/" + tag_filename, channel: tag}
        json_list.append(record)
        logger.info(f"json record: {record}")
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(json_list, f, ensure_ascii=False, indent=4)
        return tag_filename

    def select_item_based_on_ratio(self):
        # calculate the ratio of each channel
        total = self.yellow_cnt + self.blue_cnt + self.black_cnt + self.red_cnt
        total = max(total, 1)
        channels = ["yellow", "blue", "black", "red"]
        ratios = [self.yellow_cnt / total, self.blue_cnt / total, self.black_cnt / total, self.red_cnt / total]
        acc_ratios = [ratios[i - 1] + ratios[i] for i in range(1, 4)]
        channel = channels[-1]
        for i in range(4):
            if acc_ratios[-1] * random.random() <= acc_ratios[i]:
                channel = channels[i]
                break
        assert channel is not None, "channel is None"
        # select item based on channel randomly
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
            channel = random.choice(["blue", "black", "red"])
        if channel == "blue" and blue_items:
            selected_item = random.choice(blue_items)
            self.blue_cnt += 1
            return selected_item
        elif channel == "blue":
            channel = random.choice(["black", "red"])
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

    def process_multi_tag(self):
        self.multi_tag = True
        self.process_file()

    def process_single_tag(self):
        self.multi_tag = False
        self.process_file()

    def process_file(self):
        # if all file has been processed
        for i in range(self.processed_num, self.total_num):
            # select one item from the list
            file_item = self.select_item_based_on_ratio()
            filename = file_item.text()
            self.unprocessed_list.setCurrentItem(file_item)

            # process the file
            if self.multi_tag:
                dialog = TagAllDialog(self.filename_map[filename], self.enable_pred)
            else:
                dialog = TagDialog(self.filename_map[filename], self.enable_pred)
            result = dialog.exec()
            # remove the item from the list
            row = self.unprocessed_list.row(file_item)
            self.unprocessed_list.takeItem(row)
            self.unprocessed_list.setCurrentItem(self.unprocessed_list.item(0))

            if result == QDialog.Accepted or result == 22:
                self.unprocessed_label.setText(f"Unprocessed:{self.unprocessed_list.count()}")
                tag = dialog.get_tag() if result == QDialog.Accepted else dialog.get_auto_tag()
                assert tag != "", "tag can not be empty"
                if isinstance(tag, dict):
                    for color in tag:
                        if tag[color] == "":
                            continue
                        tag_filename = self.save_image(filename, tag[color], i, color)
                        self.processed_list.addItem(tag_filename)
                else:
                    tag_filename = self.save_image(filename, tag, i)
                    self.processed_list.addItem(tag_filename)
                self.processed_label.setText(
                    f"Processed:{self.processed_list.count()}(Total{self.processed_list.count() + self.processed_num})")
                self.progress_bar.setValue(i)
                self.update_figure()  # update the color ratio figure
                # remove the file from the origin folder
                os.remove(self.filename_map[filename])
            elif result == QDialog.Rejected:
                break
            elif result == 11:
                self.unprocessed_label.setText(f"Unprocessed:{self.unprocessed_list.count()}")
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
@click.option("--test-ratio", type=float, default=0.1)
@click.option("--enable-pred", is_flag=True)
@click.option("--multi-tag", is_flag=True)
def main(dataset_dir, output_dir, test_ratio, enable_pred, multi_tag):
    dataset_dir = root_dir / "dataset" / dataset_dir
    save_dir = root_dir / "dataset" / output_dir / "images"
    os.makedirs(save_dir, exist_ok=True)

    app = QApplication()
    app.setStyleSheet(dark_stylesheet)

    win = TagWindow(dataset_dir, save_dir, test_ratio, enable_pred, multi_tag)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
