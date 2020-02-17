import numpy as np
from functools import reduce

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

def design_matrix(x):
    fi_mat_tmp = np.array([
            list(map(lambda x:x**0,x)),
            list(map(lambda x:x**1,x)),
            list(map(lambda x:x**2,x)),
            list(map(lambda x:x**3,x)),
            list(map(lambda x:x**4,x)),
            list(map(lambda x:x**5,x)),
            list(map(lambda x:x**6,x)),
            list(map(lambda x:x**7,x)),
            list(map(lambda x:x**8,x)),
            list(map(lambda x:x**9,x))]
           )
    return fi_mat_tmp.T

def regression(data, label):
    lamda=50
    x1=np.array(list(map(lambda x:x[0],data)))
    x2=np.array(list(map(lambda x:x[1],data)))
    y=np.array(label)

    fi_mat1 = design_matrix(x1)
    fi1 = np.linalg.inv(fi_mat1.T.dot(fi_mat1)+lamda*np.identity(10)).dot(fi_mat1.T)
    w_ridge1 = fi1.dot(y)

    fi_mat2 = design_matrix(x2)
    fi2 = np.linalg.inv(fi_mat2.T.dot(fi_mat2) + lamda * np.identity(10)).dot(fi_mat2.T)
    w_ridge2 = fi2.dot(y)
    y_res = w_ridge1.dot(fi_mat1.T)+w_ridge2.dot(fi_mat2.T)
    b=y-y_res
    # print(b)
    b=reduce(lambda x,y:x+y,b)/200
    y_res = w_ridge1.dot(fi_mat1.T) + w_ridge2.dot(fi_mat2.T)+b
    yhat=list(map(lambda x:0 if x<0.5 else 1,list(y_res)))
    c=0
    for i in yhat-y:
        if i==0:
            c=c+1
    print("lambda :",lamda)
    print("训练集错误率：",1-c/200.0)
    print("最小二乘解：")
    print("w1:",w_ridge1)
    print("w2:",w_ridge1)
    print("b:",b)
    return w_ridge1,w_ridge2,b

def fi(x):
    f=np.ones((1,10))*x
    for i in range(10):
        f[0,i]=f[0,i]**i
    return f

def predict(w1,w2,b,test):
    res=list(map(lambda x:w1.dot(fi(x[0]).T)+w2.dot(fi(x[1]).T+b),test))
    yhat = list(map(lambda x: 0 if x < 0.5 else 1, res))
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

#
data, label, test, ptest, c1test = readData()
w1,w2,b=regression(data, label)
y=predict(w1,w2,b,test)
e=errorRate(ptest, c1test,y)
print("测试集错误率:",e)



