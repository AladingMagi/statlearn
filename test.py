import numpy as np
import matplotlib.pyplot as plt

X=np.array([[1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1]])#此时的X数据为四个点，用单层感知器无法进行分类


#标签
Y=np.array([[0,1,1,0]])
#权值初始化，取值范围-1到1
V=np.random.random((3,4))*2-1
W=np.random.random((4,1))*2-1
# print(V)
# print(W)
#定义学习率
lr=0.11

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

def RELU(x):
    return np.where(x>0,x,0)

def dRELU(x):
    return np.where(x>0,1,0.1)

def updata():#更新权值函数
    global X,Y,W,V,lr
    # L1=sigmoid(np.dot(X,V))#隐藏层的输出（4，4）#sigmoid作用与输出乘以权值矩阵
    # L2=sigmoid(np.dot(L1,W))#输出层的输出（4，1）#sigmoid作用与上一层输出乘以权重
    L1 = RELU(np.dot(X, V))
    L2 = RELU(np.dot(L1, W))

    #误差信号
    # #逆向传播先算L2的倒数，然后再算L1的倒数
    # L2_delta=(Y.T-L2)*dsigmoid(L2)#L2的倒数=理想输出-实际输出*L2经过激活函数的倒数
    # L1_delta=L2_delta.dot(W.T)*dsigmoid(L1)#L2的倒数=L2的倒数*权值（上一层的反馈）*L1进过激活函数的倒数
    L2_delta = (Y.T - L2) * dRELU(np.dot(L1, W))
    L1_delta = L2_delta.dot(W.T) * dRELU(np.dot(X, V))

    #权值更新W的改变与V的改变
    W_C=lr*L1.T.dot(L2_delta)#学习率*L1.T*L2的倒数
    V_C=lr*X.T.dot(L1_delta)#同理

    W=W+W_C#W+W的改变
    V=V+V_C

for i in range(20000):#迭代20000次
    updata()#更新权值
    if i%500==0:#每隔500次输出一次误差
        L1 = RELU(np.dot(X, V))
        L2 = RELU(np.dot(L1, W))
        # L1 = sigmoid(np.dot(X, V))  # 隐藏层的输出（4，4）#sigmoid作用与输出乘以权值矩阵
        # L2 = sigmoid(np.dot(L1, W))  # 输出层的输出（4，1）#sigmoid作用与上一层输出乘以权重
    print ('Error:',np.mean(np.abs(Y.T-L2)))#打印误差

#print ("L2:",L2)

#显示结果
def judge(x):
    if(x>=0.5):
        return 1
    else:
        return -1
print("输入：\n[0,0],[0,1],[1,0],[1,1]")
print("leakrelu函数作为激活函数分类结果：")
print(list(map(judge,L2)))


