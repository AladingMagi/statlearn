import numpy as np
import math
from functools import reduce
from sklearn.model_selection import KFold
def pre(x):
    x=x[:-1].split(",")
    x=map(lambda i:float(i),x)
    return list(x)
def readData():
    data_path='C:\\Users\\Administrator\\Desktop\\data1\\xtrain.txt'
    label_path='C:\\Users\\Administrator\\Desktop\\data1\\ctrain.txt'
    test_path='C:\\Users\\Administrator\\Desktop\\data1\\xtest.txt'
    ptest_path='C:\\Users\\Administrator\\Desktop\\data1\\ptest.txt'
    c1test_path='C:\\Users\\Administrator\\Desktop\\data1\\c1test.txt'

    f = open(data_path,"r")
    data = f.readlines()
    f.close()
    data=list(map(pre,data))

    f = open(label_path, "r")
    label = f.readlines()
    f.close()
    label=list(map(lambda x:float(x[:-1]),label))

    f = open(test_path, "r")
    test_data = f.readlines()
    f.close()
    test_data=list(map(pre,test_data))

    f = open(ptest_path, "r")
    ptest_data = f.readlines()
    f.close()
    ptest_data = list(map(lambda x:float(x[:-1]), ptest_data))

    f = open(c1test_path, "r")
    c1test_data = f.readlines()
    f.close()
    c1test_data = list(map(lambda x:float(x[:-1]), c1test_data))


    return data,label,test_data,ptest_data,c1test_data

def sigmoid(inx):
    if inx>=0:
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))

def fi(x,n):
    f=np.array([1]*n)*x
    for i in range(n):
        f[i]=f[i]**i
    return f

def gradient(x,y,yhat,n):
    w1=[]
    w2=[]
    x1=np.array(list(map(lambda e:e[0],x)))
    x2=np.array(list(map(lambda e:e[1],x)))
    for i in range(n):
        g=(yhat-y)*(x1**i)
        w1.append(reduce(lambda a,b:a+b,g))
    for i in range(n):
        g = (yhat-y)*(x2**i)
        w2.append(reduce(lambda a,b:a+b, g))
    return np.array(w1)/200.0,np.array(w2)/200.0

def logistic(data,label,w1,w2,n):
    eta=0.001
    w1=np.array(w1)
    w2=np.array(w2)
    err=100
    while err>0.01:
        yhat=list(map(lambda x:sigmoid(w1.dot(fi(x[0],n).T) + w2.dot(fi(x[1],n).T)), data))
        g1,g2=gradient(data, np.array(label), np.array(yhat), n)
        w1=w1-eta*g1
        w2=w2-eta*g2
        err=(reduce(lambda x, y: x + y, g1)+reduce(lambda x, y: x + y, g2))
        # print("err:",err)

    yhat = list(map(lambda x: sigmoid(w1.dot(fi(x[0], n).T) + w2.dot(fi(x[1], n).T)), data))
    yhat = list(map(lambda x: 0 if x < 0.5 else 1, list(yhat)))
    c = 0
    for i in yhat - np.array(label):
        if i == 0:
            c = c + 1
    print("训练集错误率：", 1 - c / 200.0)
    return w1,w2

def predict(w1,w2,test,n):
    w1 = np.array(w1)
    w2 = np.array(w2)
    yhat = list(map(lambda x: sigmoid(w1.dot(fi(x[0], n).T) + w2.dot(fi(x[1], n).T)), test))
    yhat = list(map(lambda x: 0 if x < 0.5 else 1, list(yhat)))
    return yhat

def errorRate(ptest, c1test,y):
    sum1=0
    sum2=0
    i=0
    for elem in y:
        if elem==0:
            sum1=sum1+ptest[i]*c1test[i]
        else:
            sum2=sum2+ptest[i]*(1-c1test[i])
        i=i+1
    return sum1+sum2
def kfold(train,label):
    train_X = None
    train_y = None
    test_X = None
    test_y = None
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(train):
        train_X, train_y = train[train_index], label[train_index]
        test_X, test_y = train[test_index], label[test_index]
    return train_X, train_y,test_X, test_y

def K5crossValid(data,label,w1,w2,n):
    train_X, train_y,test_X, test_y=kfold(np.array(data), np.array(label))
    w1,w2=logistic(train_X, train_y, w1, w2, n)
    c = 0
    yhat = predict(w1,w2,test_X)
    for j in yhat - np.array(test_y):
        if j == 0:
            c = c + 1
    print("test_X错误率：", 1 - c / len(yhat))
    print("w1:",w1)
    print("w2:",w2)
    print("阶数：", n)
    print("------------------------------------")


# for n in range(1,7):
#     w1=[1]*n
#     w2=[1]*n
#     data, label, test_feat, ptest, c1test = readData()
#     K5crossValid(data,label,w1,w2,n)
data, label, test, ptest, c1test = readData()
n=1
w1=[0.03751094]
w2=[0.03751094]
y=predict(w1,w2,test,n)
error=errorRate(ptest, c1test,y)
print("阶数：",n)
print("test_feat data error rate:",error)
print("------------------------------------")

n=2
w1=[-0.16731435 ,-0.32561446]
w2=[-0.16731435  ,0.96969812]
y=predict(w1,w2,test,n)
error=errorRate(ptest, c1test,y)
print("阶数：",n)
print("test_feat data error rate:",error)
print("------------------------------------")

n=3
w1=[-0.29362915 ,-0.16817008  ,0.02905996]
w2=[-0.29362915  ,1.10376095 ,-0.04193191]
y=predict(w1,w2,test,n)
error=errorRate(ptest, c1test,y)
print("阶数：",n)
print("test_feat data error rate:",error)
print("------------------------------------")

n=4
w1=[-0.41913861 ,-0.04081262  ,0.13242527 ,-0.05556762]
w2=[-0.41913861  ,0.97202039 ,-0.10757238  ,0.11347727]
y=predict(w1,w2,test,n)
error=errorRate(ptest, c1test,y)
print("阶数：",n)
print("test_feat data error rate:",error)
print("------------------------------------")

n=5
w1=[-0.32771255  ,0.33248298  ,0.15500923 ,-0.28473129  ,0.05529287]
w2=[-0.32771255  ,0.7810588  ,-0.1019362   ,0.37034372 ,-0.12405895]
y=predict(w1,w2,test,n)
error=errorRate(ptest, c1test,y)
print("阶数：",n)
print("test_feat data error rate:",error)
print("------------------------------------")

n=6
w1=[ 0.59353832  ,0.59783182  ,0.37382406 ,-0.18736316 ,-0.21491028  ,0.05269908]
w2=[ 0.59353832  ,0.92156586  ,0.61128726  ,0.74773815  ,0.07781243 ,-0.16409438]
y=predict(w1,w2,test,n)
error=errorRate(ptest, c1test,y)
print("阶数：",n)
print("test_feat data error rate:",error)
print("------------------------------------")
