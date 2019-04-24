# Toy-NN：基于numpy的简单神经网络组建

## 基本特性

- 可自由叠加网络层，耦合度低
- 多种激活函数（sigmoid, tanh, relu, softmax）
- 多种损失函数（cross entropy, mean square error, mean absolute error）
- 使用小批量梯度下降，可自定义`batch_size`和`epochs`
## 使用方法

- 见`main.py`。
- 博客描述：https://riroaki.github.io/ML-8-My-Neuron-Network-Model-2/

## 运行效果（mnist）

```shell
# tanh + sigmoid + softmax
# 128 / 5
100%|██████████| 469/469 [04:30<00:00,  2.37it/s]
Total loss for epoch 5: 36775.7511731053, time cost: 270.526123046875 secs.
Total loss for prediction: 6180.927977005118, accuracy: 0.8236
```

## 待补充

- adam

- 目前不允许在中间加入softmax作为激活函数的层（理论上这么做没有什么意义）