from answerMultiLayerPerceptron import buildNewGraph
import mnist
import numpy as np
import pickle
from util import setseed
from autograd.utils import PermIterator
from scipy.ndimage import rotate, shift

setseed(0) # 固定随机数种子以提高可复现性
save_path = "model/mine.npy"

X_trn = mnist.trn_X.reshape(-1, 28, 28)
Y_trn = mnist.trn_Y
X_val = mnist.val_X.reshape(-1, 28, 28)
Y_val = mnist.val_Y

# 随机选择2000个验证集样本，并与训练集合并
indices = np.random.choice(X_val.shape[0], 2000, replace=False)
selected_val_X = X_val[indices]
selected_val_Y = Y_val[indices]
X = np.concatenate((X_trn, selected_val_X), axis=0)
Y = np.concatenate((Y_trn, selected_val_Y), axis=0)

# 对合并后的数据集进行数据增强，增加样本数量，旋转角度为-15, -10, -5, 5, 10，15 ; 平移范围为-4, -2, 2, 4
X_aug = []
Y_aug = []
for x, y in zip(X, Y):
    for angle in [-15, -10, -5, 5, 10, 15]:
        X_aug.append(rotate(x, angle, reshape=False))
        Y_aug.append(y)
    for shift_x in [-4, -2, 2, 4]:
        for shift_y in [-4, -2, 2, 4]:
            X_aug.append(shift(x, (shift_x, shift_y)))
            Y_aug.append(y)
X = np.concatenate((X, X_aug), axis=0)
Y = np.concatenate((Y, Y_aug), axis=0)

# 对X进行展平
X = X.reshape(-1, 784)

# 超参数
lr = 1e-3   # 学习率
wd1 = 1e-5  # L1正则化
wd2 = 1e-4  # L2正则化
batchsize = 128
batchsize1 = 394
batchsize2 = 197
batchsize3 = 98

if __name__ == "__main__":
    graph = buildNewGraph(Y, batchsize1, batchsize2, batchsize3)
    best_train_acc = 0
    dataloader = PermIterator(X.shape[0], batchsize)
    for i in range(1, 23+1):
        hatys = []
        ys = []
        losses = []
        graph.train()
        for perm in dataloader:
            tX = X[perm]
            tY = Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losses.append(loss)
        loss = np.average(losses)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)