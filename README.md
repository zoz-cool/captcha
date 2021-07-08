# 端到端不定长验证码识别模型

## 摘要
国税网发票查验验证码识别模型

## 算法原理
- 将验证码按颜色通道分离之后识别，不识别文字。
- 使用残差网络提取特征，循环网络识别，CTC处理不定长。

## 结果

- 损失[sample num: 10000]
![损失](train_loss.svg)
- 测试集[sample num: 1000]正确率
![测试集正确率](test_accuracy.svg)
- 训练集批次[batch_size: 128]正确率
![训练集批次正确率](train_accuracy.svg)
