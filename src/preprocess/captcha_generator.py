import json
import os
import random
import pathlib
import argparse
import shutil
import sys
import time

from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import make_interp_spline


class CaptchaGenerator:
    """
    生成验证码图片
    """

    def __init__(self, font_path, font2_path, chinese_words_path, width=120, height=50, max_words=6):
        self.width = width
        self.height = height
        self.max_words = max_words
        self.font = ImageFont.truetype(font_path, 14)
        self.font2 = ImageFont.truetype(font2_path, 32)
        self.characters = [chr(i) for i in range(65, 91)]  # 大写字母
        self.characters += [str(i) for i in range(10)]  # 阿拉伯数字
        self.chinese_words = ""  # 汉字
        with open(chinese_words_path, encoding="utf-8") as f:
            self.chinese_words = f.read()

    @staticmethod
    def get_light_colors(num_colors):
        # 初始化一个空列表来存储RGB值
        light_colors = []

        # 生成指定数量的浅色RGB值
        for _ in range(num_colors):
            r = random.randint(200, 255)
            g = random.randint(200, 255)
            b = random.randint(200, 255)
            light_colors.append((r, g, b))

        return light_colors

    @staticmethod
    def get_dark_colors(num_colors):
        # 初始化一个空列表来存储RGB值
        dark_colors = []

        # 生成指定数量的深色RGB值
        for _ in range(num_colors):
            r = random.randint(100, 150)
            g = random.randint(100, 150)
            b = random.randint(100, 150)
            dark_colors.append((r, g, b))

        return dark_colors

    def gen_next(self, min_num=4, max_num=6):
        assert min_num <= max_num <= self.max_words, "不能超出最大字符数"
        bg_colors = self.get_light_colors(1)
        img = Image.new('RGBA', (self.width, self.height), color=bg_colors[0])
        self.draw_river(img)
        self.draw_line(img)
        label = self.draw_text_rotate(img, random.randint(min_num, max_num))
        return img, label

    def draw_river(self, img: Image.Image):
        draw = ImageDraw.Draw(img)

        bg_colors = self.get_dark_colors(1)
        # 创建两条平行的样条曲线
        xs = np.linspace(0, self.width, num=500)
        y1 = 15 * np.sin(2 * np.pi * xs / (3 * self.width) + np.random.uniform(-np.pi, np.pi))
        y2 = y1 + self.height  # 修改这里使曲线更宽

        # 使用scipy的make_interp_spline函数生成样条曲线
        spl1 = make_interp_spline(xs, y1)
        spl2 = make_interp_spline(xs, y2)
        xnew = np.linspace(0, self.width, num=1000)
        y1new = spl1(xnew)
        y2new = spl2(xnew)

        # 将两条曲线之间的区域填充颜色，设置颜色为淡绿色
        points = list(zip(np.concatenate([xnew, xnew[::-1]]), np.concatenate([y1new, y2new[::-1]])))
        draw.polygon(points, fill=bg_colors[0])
        # 添加带颜色的噪点
        for _ in range(200):  # 添加1000个噪点
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # 随机颜色
            draw.point((x, y), fill=color)

    @staticmethod
    def draw_line(img: Image.Image, color=(0, 255, 0)):
        draw = ImageDraw.Draw(img)

        width, height = img.size
        num = random.randint(0, 2)
        # 添加两条随机线段
        for _ in range(num):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=1)

    @staticmethod
    def shift_image(img: Image.Image, dx: int, dy: int) -> Image.Image:
        """
        平移图片
        :param img: PIL Image对象
        :param dx: 水平平移距离，正值向右，负值向左
        :param dy: 垂直平移距离，正值向下，负值向上
        :return: 平移后的PIL Image对象
        """
        width, height = img.size
        # 使用Image.transform函数进行平移
        new_img = img.transform((width, height), Image.AFFINE, (1, 0, dx, 0, 1, dy))
        return new_img

    def draw_text_rotate(self, img: Image.Image, num, ratio=0.4) -> dict:
        """
        若要添加字样，则设为true，数字默认开启
        :param ratio: 汉字出现概率
        :param img:
        :param num: 验证码字符个数
        :return: label
        """

        colors = {(0, 0, 0): "black", (255, 0, 0): "red", (0, 0, 255): "blue", (255, 255, 0): "yellow"}  # 随机颜色
        label = []  # 存储label: 类型， 0-->字母或数字，1-->汉字

        word_width = int(self.width / self.max_words)
        # 当前字符的起始位置，尽量保证居中
        prefix = word_width * (self.max_words - num) / 2
        if random.random() < ratio:
            # 汉字随机生成1~3个
            chinese_num = random.randint(1, min(3, (num + 1) // 2))
            for i in range(chinese_num):
                # 随机选择一个汉字
                text = random.choice(self.chinese_words)
                label.append([text, 1])
        for _ in range(len(label), num):
            # 随机选择一个字母或数字
            text = random.choice(self.characters)
            label.append([text, 0])
        for i in range(num):
            text, t = label[num - 1 - i]
            img0 = Image.new('RGBA', img.size, (255, 255, 255, 0))
            draw0 = ImageDraw.Draw(img0, mode='RGBA')
            # 放在中间
            x = self.width / 2 - self.font.size / 2
            y = self.height / 2 - self.font.size
            color = random.choice(list(colors.keys()))
            label[num - 1 - i].append(color)
            draw0.text((x, y), text, color, font=self.font2 if t > 0 else self.font)
            # 获得一个旋转字
            img1 = img0.rotate(random.uniform(-50, 50), Image.BILINEAR, expand=False)
            # 平移
            shift_x = int(prefix + word_width / 2 - self.width / 2 + word_width * i)
            img2 = self.shift_image(img1, shift_x, random.randint(-6, 6))
            # Crop img1 to the same size as img
            img.paste(Image.alpha_composite(img, img2))
        # 按照color分组
        label_map = {}
        for text, _, c in label:
            key = colors[c]
            label_map[key] = label_map.get(key, "") + text
        label_map["text"] = "".join([text for text, _, _ in label])
        return label_map


def parse_args():
    proj_dir = pathlib.Path(__file__).parent.parent.parent.absolute()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--min_num', type=int, default=4)
    parser.add_argument('--max_num', type=int, default=6)
    parser.add_argument('--output', type=pathlib.Path, default=proj_dir / "dataset/generated")
    parser.add_argument('--test_ratio', type=float, default=0.4)
    return parser.parse_args(sys.argv[1:])


def batch_save(imgs, labels, output_dir: pathlib.Path, test_ratio=0.4, index=0):
    label_file = "train.json" if random.random() > test_ratio else "test.json"
    image_dir = output_dir.absolute() / "images"
    os.makedirs(image_dir, exist_ok=True)
    for i, img in enumerate(imgs):
        save_path = image_dir / f"{index + i:0>7d}-{labels[i]['text']}.png"
        img.save(save_path)
    json_file = output_dir / label_file
    file_records = []
    if os.path.isfile(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            file_records = json.load(f)
    for i, label in enumerate(labels):
        label["index"] = index + i
        label["path"] = "images/" + f"{index + i:0>7d}-{labels[i]['text']}.png"
        file_records.append(label)
    with open(json_file, "w", encoding="utf-8") as fout:
        json.dump(file_records, fout, ensure_ascii=False, indent=4)


def main():
    proj_dir = pathlib.Path(__file__).parent.parent.parent.absolute()
    assets_dir = proj_dir / "assets"
    font_path = str(assets_dir / "font/3D-Hand-Drawns-1.ttf")
    font2_path = str(assets_dir / "font/HanYiFangSongJian-1.ttf")
    chinese_words_path = str(assets_dir / "chinese_words.txt")

    args = parse_args()
    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    gen = CaptchaGenerator(font_path, font2_path, chinese_words_path, width=120, height=50, max_words=args.max_num)
    batch_imgs = []
    batch_labels = []
    batch_size = 1000
    tbar = tqdm(range(args.num))
    batch_count = 0
    for i in tbar:
        img, label = gen.gen_next(min_num=args.min_num, max_num=args.max_num)
        batch_imgs.append(img)
        batch_labels.append(label)
        tbar.set_description(f"index={i + 1}")
        if (i > 0 and i % batch_size == 0) or i == args.num - 1:
            tbar.set_description(f"index={i + 1}, batch save!")
            batch_save(batch_imgs, batch_labels, args.output, test_ratio=args.test_ratio,
                       index=batch_size * batch_count)
            batch_imgs = []
            batch_labels = []
            batch_count += 1


if __name__ == '__main__':
    main()
