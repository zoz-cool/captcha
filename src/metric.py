#! -*- coding: utf-8 -*-

import paddle
from paddle.metric import Metric

import decoder


class WordsPrecision(Metric):
    """单词正确率"""

    def __init__(self, vocabulary, name='words-precision'):
        super().__init__()
        self.vocabulary = vocabulary
        self.correct = 0
        self.all = 0
        self._name = name

    def update(self, outputs, labels):
        """更新统计指标"""
        if not isinstance(outputs, paddle.Tensor):
            outputs = paddle.to_tensor(outputs)
        # 解码获取识别结果
        for output, label in zip(outputs, labels):
            pred_text = decoder.ctc_greedy_decoder(output, self.vocabulary)
            label_text = decoder.label_to_string(label, self.vocabulary)
            # 计算字错率
            cer = decoder.cer(pred_text, label_text)
            self.correct += max(0, len(label_text) - cer)
            self.all += len(label_text)

    def reset(self):
        """
        Resets all the metric state.
        """
        self.correct = 0
        self.all = 0

    def accumulate(self):
        """
        Calculate the final precision.

        Returns:
            A scaler float: results of the calculated precision.
        """
        return float(self.correct) / self.all if self.all != 0 else .0

    def name(self):
        """
        Returns metric name
        """
        return self._name
