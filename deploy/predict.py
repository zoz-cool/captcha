import io
from typing import List
from itertools import groupby

import numpy as np
from PIL import Image
from ppqi import InferenceModel
from fastapi import UploadFile, File, FastAPI


app = FastAPI()

# 加载模型
model = InferenceModel(
    modelpath='inference/model', 
    use_gpu=False, 
    use_mkldnn=True
)
model.eval()

with open("vocabulary.txt", 'r', encoding='utf-8') as fin:
    vocabulary = fin.readlines()
vocabulary = [w.strip() for w in vocabulary if w.strip()]


@app.post("/captcha")
async def upload_images(files: List[UploadFile] = File(...)):
    imgs = []
    for file in files:
        contents = await file.read()
        # 将文件内容转换为字节流
        image_stream = io.BytesIO(contents)
        # 使用PIL库读取图片数据
        pil_image = Image.open(image_stream).convert("RGB")
        img_arr = preprocess(pil_image)
        imgs.append(img_arr)

    labels = make_pred(np.array(imgs).astype("float32"))
    ans = [{"filename": file.filename, "label": label} for file, label in zip(files, labels)]
    return {"data": ans}



def preprocess(img: Image):
    # 将PIL Image对象转换为numpy数组
    img_arr = np.array(img).astype("float32") / 255.0
    
    # 归一化
    mean = np.array([0.55456273, 0.5225813,  0.51677391])
    std = np.array([0.23375643, 0.23862716, 0.23951546])
    img_arr = normalize(img_arr, mean, std)
    
    # RGB --> BGR
    img_arr = img_arr[:,:,::-1]
    
    # 转换为NCHW格式
    img_arr = np.transpose(img_arr, (2, 0, 1))
    
    return img_arr


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def normalize(image, mean, std):
    image = image.astype(np.float32)
    image = (image - mean) / std
    return image


def ctc_greedy_decoder(probs_seq, vocabulary):
    """CTC贪婪（最佳路径）解码器。
    由最可能的令牌组成的路径被进一步后处理
    删除连续的重复和所有的空白。
    :param probs_seq: 每个词汇表上概率的二维列表字符。
                      每个元素都是浮点概率列表为一个字符。
    :type probs_seq: list
    :param vocabulary: 词汇表
    :type vocabulary: list
    :return: 解码结果字符串
    :rtype: baseline
    """
    # 尺寸验证
    for probs in probs_seq:
        if not len(probs) == len(vocabulary) + 1:
            raise ValueError("probs_seq 尺寸与词汇不匹配")
    # argmax以获得每个时间步长的最佳指标
    max_index_list = np.argmax(probs_seq, -1)
    # 删除连续的重复索引
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    # 删除空白索引
    blank_index = len(vocabulary)
    index_list = [index for index in index_list if index != blank_index]
    # 将索引列表转换为字符串
    return ''.join([vocabulary[index] for index in index_list])


def make_pred(imgs):
    outputs = softmax(model(imgs))
    # 解码获取识别结果
    pred_list = []
    for output in outputs:
        pred = ctc_greedy_decoder(output, vocabulary)
        pred_list.append(pred)
    return pred_list
