#! -*- coding: utf-8 -*-

import paddle


class MultiChannelCTCLoss(paddle.nn.Layer):
    def __init__(self, num_classes, max_len, channel_num=5):
        super().__init__()
        self.max_len = max_len
        self.channel_num = channel_num

        self.ctc_loss = paddle.nn.CTCLoss(blank=num_classes)

    def forward(self, inputs: paddle.Tensor, labels: paddle.Tensor):
        in_lengths = paddle.full(shape=[inputs.shape[1]], fill_value=inputs.shape[0] // self.channel_num, dtype="int64")
        total_loss = paddle.zeros(shape=[1], dtype="float32")
        channel_len = labels.shape[1] // self.channel_num
        for channel_id in range(0, self.channel_num, channel_len):
            channel_labels = labels[:, channel_id:channel_id + channel_len]
            channel_inputs = inputs[channel_id:channel_id + channel_len, :, :]
            label_lengths = paddle.sum(channel_labels != -1, axis=-1, dtype="int64")
            channel_loss = self.ctc_loss(channel_inputs, channel_labels, in_lengths, label_lengths)

            total_loss += channel_loss
        return total_loss
