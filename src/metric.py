#! -*- coding: utf-8 -*-

import paddle
from paddle.metric import Metric

import decoder


class WordsPrecision(Metric):
    """单词正确率"""

    def __init__(self, vocabulary, name='words-precision'):
        super().__init__()
        self.vocabulary = vocabulary
        self.cer_value = 0
        self.cer_count = 0
        self._name = name

    def update(self, outputs, labels):
        """更新统计指标"""
        if not isinstance(outputs, paddle.Tensor):
            outputs = paddle.to_tensor(outputs)
        outputs = paddle.nn.functional.softmax(outputs, axis=-1).transpose([1, 0, 2])
        # 解码获取识别结果
        for output, label in zip(outputs, labels):
            pred_text = decoder.ctc_greedy_decoder(output, self.vocabulary)
            label_text = decoder.label_to_string(label, self.vocabulary)
            # 计算字错率
            self.cer_value += decoder.cer(pred_text, label_text) / float(len(label_text))
            self.cer_count += 1

    def reset(self):
        """
        Resets all the metric state.
        """
        self.cer_value = 0
        self.cer_count = 0

    def accumulate(self):
        """
        Calculate the final precision.

        Returns:
            A scaler float: results of the calculated precision.
        """
        return self.cer_value / max(1, self.cer_count)

    def name(self):
        """
        Returns metric name
        """
        return self._name
