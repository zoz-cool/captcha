"""支持工具类"""

import torch
from .config import config


def decode_output(outputs):
    """解码输出"""
    output = torch.argmax(outputs, dim=-1)
    output = output.permute(1, 0)
    preds = []
    for predict in output:
        predict = torch.unique_consecutive(predict)
        predict = predict[predict != config.num_classes]  # 去除占位符
        pred = ''.join(config.classes[i.item()] for i in predict)
        preds.append(pred)
    return preds


def decode_labels(labels, label_length):
    """解码输入标签"""
    label_list = []
    start = 0
    for length in label_length:
        label = labels[start:start+length]
        start += length.item()
        lab = ''.join(config.classes[i.item()] for i in label)
        label_list.append(lab)
    return label_list


def calculate_acc(output, target, target_length, call_label=False):
    """计算正确率，call_label=False时计算标签数以及预算正确数，；否则计算真实标签及预测标签"""
    preds = decode_output(output)
    labels = decode_labels(target, target_length)

    if call_label:
        return preds, labels
    else:
        correct_num = 0
        total_num = len(preds)
        for pred, label in zip(preds, labels):
            if pred == label:
                correct_num += 1
        return correct_num, total_num

