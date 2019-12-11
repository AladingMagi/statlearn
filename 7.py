import random
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
T=3
class Node:
    left=None
    right=None
    x=None
    y=None
    resx1=None
    resy1=None
    resx2 = None
    resy2 = None
    height=None
    split=1

    def __init__(self,height,x,y):
        self.height=height
        self.x=x
        self.y=y

    def newNode(self):
        self.left=Node(self.height+1,self.x[0:self.split],y[0:self.split])
        self.right=Node(self.height+1,self.x[self.split:len(self.x)],y[self.split:len(self.y)])

    def travel(self):
        print("height:",self.height)
        print("x:", self.x)
        print("y:",self.y)
        if self.left==None and self.right==None:
            return
        self.left.travel()
        self.right.travel()

    def train(self):
        err=[]
        for i in range(len(x)):
            tempx1=self.x[0:i]
            tempy1=self.y[0:i]
            if len(tempx1)==0 or len(tempy1)==0:
                continue
            tempx2 = self.x[i:len(x)]
            tempy2 = self.y[i:len(y)]
            if len(tempx2) == 0 or len(tempy2)==0:
                continue
            r1,e1=self.regression(tempx1,tempy1)
            r2,e2=self.regression(tempx2,tempy2)
            err.append(e1+e2)
        print("误差列表",np.array(err))
        self.split=err.index(min(err))+1

        self.resx1=self.x[0:self.split]
        self.resy1,rese1=self.regression(self.x[0:self.split],self.y[0:self.split])

        self.resx2=self.x[self.split:len(self.x)]
        self.resy2,rese2=self.regression(self.x[self.split:len(self.x)],self.y[self.split:len(self.x)])
        print("分割点index：",self.split)

    def regression(self,x,y):
        lamda = 0.1
        fi_mat = self.design_matrix(x)
        fi = np.linalg.inv(fi_mat.T.dot(fi_mat) + lamda * np.identity(4)).dot(fi_mat.T)
        w_ridge = fi.dot(y)
        y_res = w_ridge.dot(fi_mat.T)
        err = reduce(lambda x,y:x+y,map(lambda x:x**2,y_res-y))
        return list(y_res),err

    def design_matrix(self,x):
        fi_mat_tmp = np.array([
            list(map(lambda x: x ** 0, x)),
            list(map(lambda x: x ** 1, x)),
            list(map(lambda x: x ** 2, x)),
            list(map(lambda x: x ** 3, x))]
        )
        return fi_mat_tmp.T


def create_data():
    sum = np.zeros(25)
    x = np.linspace(0, 24, 25) * 0.041
    for i in range(100):
        e=np.array([random.gauss(0,0.3) for i in range(25)])
        y = np.sin(2*np.pi*x)+e
        sum+=y
    y = np.array(list(map(lambda x:x/100,sum)))
    y_origin = np.sin(2*np.pi*x)
    return x,y,y_origin
def crateTree(x,y,y_origin):
    root=Node(1,x,y)
    cur=root
    root.train()
    cur.newNode()
    # root.travel()

    cur.left.train()
    cur.right.train()
    # cur.left.travel()
    # cur.right.travel()
    print("叶节点1输入:",cur.left.resx1)
    print("叶节点1输出:", cur.left.resy1)
    print("叶节点2输入:", cur.left.resx2)
    print("叶节点2输出:", np.array(cur.left.resy2))
    print("叶节点3输入:", cur.right.resx1)
    print("叶节点3输出:", cur.right.resy1)
    print("叶节点4输入:", cur.right.resx2)
    print("叶节点4输出:", cur.right.resy2)
    resx=list(cur.left.resx1)+list(cur.left.resx2)+list(cur.right.resx1)+ list(cur.right.resx2)
    resy= list(cur.left.resy1)+list(cur.left.resy2)+list(cur.right.resy1)+list(cur.right.resy2)
    print(resx)
    print(resy)
    plt.plot(x, y, 'b', marker='o', label='origin')
    plt.plot(resx, resy, 'g', marker='^', label='result')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x, y, y_origin =create_data()
    print(x)
    print(y)
    crateTree(x, y,y_origin)


