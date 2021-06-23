"""
给图片打标签
"""

import os
import wx
from PIL import Image
import numpy as np
import shutil
from random import random
from model.main import Model



class ViewDialog(wx.Frame):
    def __init__(self, work_path, dataset_path):
        super().__init__(None, title='打验证码标签')
        panel = wx.Panel(self)    
        self.work_path = work_path
        self.dataset_path = dataset_path

        bit = wx.StaticBitmap(panel, bitmap=wx.Bitmap.FromBuffer(w1, h1, img))
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(bit, flag=wx.ALL|wx.EXPAND, border=20)
        panel.SetSizer(sizer)
        bit.Bind(wx.EVT_RIGHT_DOWN, self.on_click)
        self.SetMinSize((800, 500))
        panel.Fit()
        self.Fit()
        self.Centre()
        self.SetFocus()
        self.Bind(wx.EVT_KEY_DOWN, self.press_key)
        wx.CallLater(10, self.on_click, None)

    def press_key(self, e):
        if e.GetKeyCode() == 32:
            self.on_click(None)
        elif e.GetKeyCode() == 13:
            self.Destroy()

    def on_click(self, e):
        if not os.path.isdir(self.out):
            os.makedirs(self.out)
        nid = len(os.listdir(self.out)) + 1
        
        m = Model()
        label = m.predict(Image.open(self.img_path))
        if not label:
            self.Destroy()
        else:    
            dlg = wx.TextEntryDialog(self, f"[{nid}] 请输入图片中的字符:", "标注字符内容", value=label)
            if dlg.ShowModal() == wx.ID_OK:
                message = dlg.GetValue()
                if len(message) == 0:
                    self.Destroy()  
                else:                
                    Image.fromarray(self.img).save(f'{self.out}/{message}_{nid}.png')
                    self.Destroy()
            dlg.Destroy()


def main(work_dir, dataset_dir):
    app = wx.App()
    dlg = ViewDialog(work_dir, dataset_dir)
    dlg.Show()
    app.MainLoop()


if __name__ == '__main__':
    work_path = 'dataset/captcha-images'
    dataset_path = 'dataset/captcha'
    main(work_path, dataset_path)

    
    