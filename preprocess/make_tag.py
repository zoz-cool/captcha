"""
给图片打标签
"""
import os
import pathlib
import random
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from queue import Queue

import click
import numpy as np
import requests
import wx
from PIL import Image

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


class MakeTagDialog(wx.Frame):
    def __init__(self, files_dir, target_dir, test_ratio=0.3):
        super().__init__(None, title='打验证码标签')
        self.panel = wx.Panel(self)
        self.files = glob(f"{files_dir}/*.png")
        self.target_dir = pathlib.Path(target_dir)
        self.exist_count = len(glob(f"{target_dir}/**/*.png"))
        self.test_ratio = test_ratio
        os.makedirs(self.target_dir / 'train', exist_ok=True)
        os.makedirs(self.target_dir / 'test', exist_ok=True)
        
        self.img_iter = self.get_img_iter()
        
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
        wx.CallLater(10, self.on_click, None)

    def get_img_iter(self):
        for i, filepath in enumerate(self.files):
            _, label, ci = get_pred_label(filepath)
            yield i+1, label, ci, filepath

    def next_img(self):        
        try:
            idx, label, ci, file_path = next(self.img_iter)
            img = Image.open(file_path)
            bitmap = wx.Bitmap.FromBuffer(img.size[0], img.size[1], np.array(img))
            self.bitmap.SetBitmap(bitmap)
            dlg = wx.TextEntryDialog(self, f"[{idx}/{len(self.files)}]"
                                           f"请输入图片中的字符:(ci={ci})", "标注字符内容", value=label)
            dst = 'train' if random.random() > self.test_ratio else 'test'
            save_path = self.target_dir / dst / f"{label}_{self.exist_count+idx}.png"
            if dlg.ShowModal() == wx.ID_OK:
                message = dlg.GetValue()
                if len(message) > 0:
                    shutil.move(file_path, save_path)
                dlg.Destroy()
            else:
                dlg.Destroy()
        except Exception:
            print(traceback.format_exc())
            return True

    def on_click(self, e):
        status = self.next_img()
        if status:
            self.Destroy()
        else:
            self.on_click(None)


def manual_tag(files_dir, target_dir):
    """make tag manually"""
    app = wx.App()
    dlg = MakeTagDialog(files_dir, target_dir)
    dlg.Show()
    app.MainLoop()


def auto_tag(files_dir, target_dir, hard_dir, min_ci=0.90, test_ratio=0.3):
    """make tag automatically"""
    os.makedirs(f"{target_dir}/train", exist_ok=True)
    os.makedirs(f"{target_dir}/test", exist_ok=True)
    os.makedirs(hard_dir, exist_ok=True)
    ori_files = glob(f"{files_dir}/*.png")
    ori_count = len(ori_files)
    count = len(glob(f"{target_dir}/**/*.png"))
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
                data_dir = 'train' if random.random() > test_ratio else 'test'
                save_path = pathlib.Path(target_dir, data_dir, f'{label}_auto_ci{ci}_{count+process_count}.png')
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
        for file_path in glob(f"{data_dir}/**/*.png", recursive=True):
            label = file_path.split('/')[-1].split('_')[0]
            fp.write(f'{file_path}\t{label}\n')


@click.command()
@click.option('--mode', required=True, type=click.Choice(['auto-tag', 'man-tag', 'gen-label']))
def run(mode):
    if mode == 'auto-tag':
        splitted_dir = root_dir / 'dataset/splitted'
        target_dir = root_dir / 'dataset/labeled'
        hard_dir = root_dir / 'dataset/hard'
        auto_tag(splitted_dir, target_dir, hard_dir)
    elif mode == 'man-tag':
        splitted_dir = root_dir / 'dataset/hard'
        target_dir = root_dir / 'dataset/labeled'
        manual_tag(splitted_dir, target_dir)
    elif mode == 'gen-label':
        prefix = root_dir / 'dataset/labeled'
        print(f'Generate train label file in {prefix}/gt_train.txt')
        generate_label_file(f'{prefix}/train', f'{prefix}/gt_train.txt')
        print(f'Generate test label file in {prefix}/gt_test.txt')
        generate_label_file(f'{prefix}/test', f'{prefix}/gt_test.txt')


if __name__ == '__main__':
    run()
