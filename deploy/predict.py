import io
from typing import List
from itertools import groupby

import numpy as np
from PIL import Image
from loguru import logger
from ppqi import InferenceModel
from fastapi import UploadFile, File, FastAPI, Form, Request

app = FastAPI()
logger.add("logs/visit.log", rotation="10 MB", encoding="utf-8", enqueue=True, compression="zip", retention="100 days")

# 加载模型
logger.info("load model from inference/model...")
model = InferenceModel(
    modelpath='inference/model',
    use_gpu=False,
    use_mkldnn=True
)
model.eval()

logger.info("load vocabulary from vocabulary.txt...")
with open("vocabulary.txt", 'r', encoding='utf-8') as fin:
    vocabulary = fin.readlines()
vocabulary = [w.strip() for w in vocabulary if w.strip()]
logger.info(f"vocabulary: {len(vocabulary)}")
channel_candidates = ["red", "blue", "black", "yellow", "text", "random"]
logger.info(f"channel candidates: {channel_candidates}")


@app.post("/captcha/predict")
async def upload_images(request: Request, channels: str = Form(...), files: List[UploadFile] = File(...)):
    channels = channels.split(",")
    filenames = [file.filename for file in files]
    host = request.client.host
    logger.info(f"[{host}] visit /captcha/predict, channels: {channels}, file num: {len(files)}, files: {filenames}")
    if len(channels) != len(files):
        return {"error": "channels must be equal to files"}
    for channel in channels:
        if channel not in channel_candidates:
            return {"error": f"channel must be one of {channel_candidates}"}
    imgs = []
    color_idx = []
    for channel, file in zip(channels, files):
        contents = await file.read()
        # 将文件内容转换为字节流
        image_stream = io.BytesIO(contents)
        # 使用PIL库读取图片数据
        pil_image = Image.open(image_stream).convert("RGB")
        img_arr, c_idx = preprocess(pil_image, channel)
        imgs.append(img_arr)
        color_idx.append(c_idx)
    labels = make_pred(np.array(imgs).astype("float32"), np.array(color_idx).astype("float32"))
    ans = [{"filename": file.filename, "label": label} for file, label in zip(files, labels)]
    logger.info(f"predicts: {ans}")
    return {"data": ans}


def preprocess(img: Image, channel: str):
    # 将PIL Image对象转换为numpy数组
    img_arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    # 归一化
    std = np.array([0.23375643, 0.23862716, 0.23951546])
    mean = np.array([0.55456273, 0.5225813, 0.51677391])
    img_arr = normalize(img_arr, mean, std)
    # 加入颜色信息
    color_index = 1.0 * channel_candidates.index(channel) / len(channel_candidates)

    return img_arr, color_index


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def normalize(image, mean, std):
    image = image.astype(np.float32)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image


def ctc_greedy_decoder(probs_seq, voc):
    """CTC贪婪（最佳路径）解码器"""
    # 尺寸验证
    for probs in probs_seq:
        if not len(probs) == len(voc) + 1:
            raise ValueError("probs_seq 尺寸与词汇不匹配")
    # argmax以获得每个时间步长的最佳指标
    max_index_list = np.argmax(probs_seq, -1)
    # 删除连续的重复索引
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    # 删除空白索引
    blank_index = len(voc)
    index_list = [index for index in index_list if index != blank_index]
    # 将索引列表转换为字符串
    return ''.join([voc[index] for index in index_list])


def make_pred(imgs: np.ndarray, color_idx: np.ndarray):
    outputs = softmax(model(imgs, color_idx))
    # 解码获取识别结果
    pred_list = []
    for output in outputs:
        pred = ctc_greedy_decoder(output, vocabulary)
        pred_list.append(pred)
    return pred_list
