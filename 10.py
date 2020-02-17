import numpy as np

x = np.array([[0.5], [1]])
t = np.array([1])
y = np.array([[0], [0]])
h1 = np.array([[0], [0]])
w1 = np.array([[0.1, 0.3], [0.2, 0.4]])
w2 = np.array([0.6, 0.8])
bias1 = np.array([[0.5], [0.5]])
bias2 = np.array([[1], [1]])


# 激活函数sigmoid
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def RELU(x):
    return x if(x>0) else 0

def QianKui():
    global h1, y
    h1 = sigmoid(np.matmul(w1, x) + bias1)
    y = sigmoid(np.matmul(w2, h1) + bias2)
    # print(h1,y)


def BackPropagation():
    global h1, y, w1, w2, bias1, bias2
    # 第二层
    delte_2 = (y - t) * y * (1 - y)  # δ2--第二层神经元的梯度系数；np.array的“*”就是Hadamard乘积
    delte_bias2 = delte_2  # Δb2
    delte_w2 = np.matmul(delte_2, h1.T)  # Δw2
    # print("δ2,Δw2:",delte_2,delte_w2)
    # print("delte_2,delte_w2:",delte_2,delte_w2)

    # 第一层
    sigma_logit_1 = h1 * (1 - h1)  # σ(logit_1)函数的导数
    delte_1 = np.matmul(w2.T, delte_2) * sigma_logit_1  # δ1--第二层神经元的梯度系数；np.array的“*”就是Hadamard乘积
    delte_bias1 = delte_1
    delte_w1 = np.matmul(delte_1, x.T)
    # print("δ1,Δw1:",delte_1,delte_w1)
    # print("delte_1,delte_w1:",delte_1,delte_w1)

    # 更新各层权重
    w1 = w1 - delte_w1
    bias1 = bias1 - delte_bias1
    # print("project3--w1&bias1:",w1,bias1)

    w2 = w2 - delte_w2
    bias2 = bias2 - delte_bias2
    # print("project3--w2&bias2:",w2,bias2)


QianKui()
print(h1, y)
BackPropagation()

for i in range(100000):
    QianKui()
    # print(y)
    BackPropagation()
    # print('........................')

print(y)
print(w1)
print(bias1)
print(w2, bias2)
