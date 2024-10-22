from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 20     # 树的数量
ratio_data = 0.95   # 采样的数据比例
ratio_feat = 0.5 # 采样的特征比例
hyperparams = {
    "depth":6, 
    "purity_bound":1e-2,
    "gainfunc": negginiDA
}

def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    trees = []
    while len(trees) < num_tree:
        # 样本扰动
        sample_mask = np.random.randint(0, 1000, 1000)
        X_ = X[sample_mask]
        Y_ = Y[sample_mask]
        tree = buildNewTree(X_, Y_, list(range(784)), hyperparams["depth"], hyperparams["purity_bound"], hyperparams["gainfunc"])
        trees.append(tree)
    return trees
    raise NotImplementedError    

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
