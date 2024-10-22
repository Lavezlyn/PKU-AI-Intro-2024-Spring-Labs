import numpy as np
import math
# 防止溢出
max_value = np.log(np.finfo(np.float64).max)
EPS = 1e-6

# 超参数
# TODO: You can change the hyperparameters here
lr = 0.12 # 学习率
wd = 0.01  # l2正则化项系数

def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    return np.dot(X, weight) + bias

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    n,d = X.shape
    # forward
    linear = predict(X, weight, bias)
    haty = sigmoid(linear) - 0.5
    # backward
    def compute_loss(prediction, y):
        if prediction*y > 500:
            return math.log(1 + math.exp(-prediction*y)) + prediction*y
        else:
            return math.log(1 + math.exp(prediction*y))
    loss = (np.sum([compute_loss(haty[i], Y[i]) for i in range(n)])) / n + wd * np.sum(weight**2)
    dweight = np.zeros(d)
    scaler = 1 - sigmoid(linear*Y)
    for i in range(n):
        dweight += scaler[i] * Y[i] * X[i]
    dweight = -dweight / n + 2 * wd * weight
    dbias = -np.sum(scaler) / n
    # update
    weight = weight - lr * dweight
    bias = bias - lr * dbias
    return haty, loss, weight, bias
    
    
        
    
