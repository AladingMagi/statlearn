import numpy as np
import math
from functools import reduce
import matplotlib.pyplot as plt
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

def distance(x,y):
    return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

def mean(l):
     res=reduce(lambda x,y:[x[0]+y[0],x[1]+y[1]],l)
     return list(map(lambda x:x/len(l),res))

def kmeans(data,k,n):
    c=[]
    clazz=[]
    for i in range(k):
        c.append(data[n*i+1])
        clazz.append([])
    flag=True
    while flag:
        for elem in data:
            d=list(map(lambda x:distance(elem,x),c))
            index=d.index(min(d))
            clazz[index].append(elem)

        flag=False
        for i in range(k):
            cnew=mean(clazz[i])
            if (c[i][0]-cnew[0])>0.01\
                    or (c[i][0]-cnew[0])>0.01:
                flag=True
            c[i]=cnew
    return c,clazz

def draw(data,c):
    color=['#00FFFF','#FAA460','#FFF8DC','#FFFF00','#8FBC8F','#696969','#FFFAF0','#CD5C5C']
    label = 0
    for clazz in data:
        for elem in clazz:
            plt.scatter(elem[0], elem[1], c=color[label], alpha=0.4)
        label = label + 1
    for ci in c:
        plt.scatter(ci[0], ci[1],  c=color[-1], alpha=0.4,marker='x',label='center')
    plt.legend()
    plt.show()

def merge(data,label,c):
    clazz=[]
    cnt=0
    for i in range(len(data)):
        d = list(map(lambda x: distance(data[i], x), c))
        index = d.index(min(d))
        if index==label[i]:
            cnt=cnt+1
        clazz.append(index)
    print("训练集误差：",1-cnt/len(data))

def predict(test,c):
    clazz = []
    for i in range(len(test)):
        d = list(map(lambda x: distance(test[i], x), c))
        index = d.index(min(d))
        clazz.append(index)
    return clazz

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

data, label, test, ptest, c1test = readData()
c1,clazz1=kmeans(data,5,5)
draw(clazz1,c1)
c2,clazz2=kmeans(c1,2,2)
draw(clazz2,c2)
merge(data,label,c2)
y=predict(test,c2)
err=errorRate(ptest, c1test,y)
print("test_feat data error rate:",err)
