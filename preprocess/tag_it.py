"""
给图片打标签
"""

import os
import wx
import numpy as np
from PIL import Image
from random import random
from model.main import Model
from preprocess.split_rgb import split_channel
import shutil


class ViewDialog(wx.Frame):
    def __init__(self, work_dir, dataset_dir):
        super().__init__(None, title='打验证码标签')
        self.panel = wx.Panel(self)
        self.work_path = work_dir
        self.files = [os.path.join(self.work_path, file_) for file_ in os.listdir(self.work_path) if
                      '.used' not in file_]
        self.dataset_path = dataset_dir
        self.train_dir = os.path.join(self.dataset_path, 'train')
        self.test_dir = os.path.join(self.dataset_path, 'test')
        if not os.path.isdir(self.train_dir):
            os.makedirs(self.train_dir)
        if not os.path.isdir(self.test_dir):
            os.makedirs(self.test_dir)
        self.num_train_files = len(os.listdir(self.train_dir)) + 1
        self.num_test_files = len(os.listdir(self.test_dir)) + 1

        self.bitmap = wx.StaticBitmap(self.panel, bitmap=wx.Bitmap(self.files[0]))
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.bitmap, flag=wx.ALL | wx.EXPAND, border=20)
        self.panel.SetSizer(sizer)
        self.bitmap.Bind(wx.EVT_RIGHT_DOWN, self.on_click)
        self.SetMinSize((800, 500))
        self.panel.Fit()
        self.Fit()
        self.Centre()
        self.SetFocus()

        self.img_iter = self.get_channel_img()
        wx.CallLater(10, self.on_click, None)

    @staticmethod
    def get_channel_img_and_predict_label(img_path):
        channel_imgs = split_channel(Image.open(img_path))
        m = Model()
        label_img = {}
        for c, img in channel_imgs.items():
            label = m.predict(img)
            if label:
                label_img[label] = img
        return label_img

    def get_channel_img(self):
        for file_path in self.files:
            channel_imgs = self.get_channel_img_and_predict_label(file_path)
            for label, img in channel_imgs.items():
                yield label, img, file_path

    def next_img(self):
        if random() > 0.3:
            output_dir = self.train_dir
            self.num_train_files += 1
            num_id = self.num_train_files
        else:
            output_dir = self.test_dir
            self.num_test_files += 1
            num_id = self.num_test_files
        try:
            label, img, file_path = next(self.img_iter)
            bitmap = wx.Bitmap.FromBuffer(*img.size, np.array(img))
            self.bitmap.SetBitmap(bitmap)
            dlg = wx.TextEntryDialog(self, f"[{num_id}/{len(self.files)*3-self.num_test_files-self.num_train_files}] "
                                           f"请输入图片中的字符:", "标注字符内容", value=label)
            if dlg.ShowModal() == wx.ID_OK:
                message = dlg.GetValue()
                if len(message) > 0:
                    img.save(f'{output_dir}/{message}_{num_id}.png')
                if os.path.isfile(file_path):
                    shutil.move(file_path, file_path.replace('.png', '.used.png'))
                dlg.Destroy()
            else:
                dlg.Destroy()
                return True
        except StopIteration:
            return True

    def on_click(self, e):
        status = self.next_img()
        if status:
            self.Destroy()
        else:
            self.on_click(None)


def main(work_dir, dataset_dir):
    app = wx.App()
    dlg = ViewDialog(work_dir, dataset_dir)
    dlg.Show()
    app.MainLoop()


if __name__ == '__main__':
    work_path = 'dataset/captcha-images'
    dataset_path = 'dataset/captcha'
    main(work_path, dataset_path)
