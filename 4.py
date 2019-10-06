import random
import numpy as np
import matplotlib.pyplot as plt

global lamda
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
def create_data100():
    y_list = []
    sum = np.zeros(25)
    x = np.linspace(0, 24, 25) * 0.041
    for i in range(100):
        e=np.array([random.gauss(0,0.3) for i in range(25)])
        y = np.sin(2*np.pi*x)+e
        y_list.append(y)
    y_origin = np.sin(2*np.pi*x)
    return x,y_list,y_origin
def design_matrix(x):
    fi_mat_tmp = np.array([
            list(map(lambda x:x**0,x)),
            list(map(lambda x:x**1,x)),
            list(map(lambda x:x**2,x)),
            list(map(lambda x:x**3,x)),
            list(map(lambda x:x**4,x)),
            list(map(lambda x:x**5,x)),
            list(map(lambda x:x**6,x)),
            list(map(lambda x:x**7,x))]
           )
    return fi_mat_tmp.T
def draw():
    lamda = 1
    x,y,y_origin = create_data()
    fi_mat = design_matrix(x)
    fi = np.linalg.inv(fi_mat.T.dot(fi_mat)+lamda*np.identity(8)).dot(fi_mat.T)
    w_ridge = fi.dot(y)
    y_res = w_ridge.dot(fi_mat.T)
    plt.plot(x, y_origin,'b',marker='o',label='y=sin(2*pi*x)')
    plt.plot(x, y,'r',marker='o',label='y=sin(2*pi*x)+e')
    plt.plot(x,y_res,'g',marker='^',label='result')
    plt.legend()
    plt.show()
def draw100():
    lamda = 0
    x,y_list,y_origin = create_data100()
    fi_mat = design_matrix(x)
    for yi in y_list:
        fi = np.linalg.inv(fi_mat.T.dot(fi_mat)+lamda*np.identity(8)).dot(fi_mat.T)
        w_ridge = fi.dot(yi)
        y_res = w_ridge.dot(fi_mat.T)
        plt.plot(x,y_res,'b',marker='^')
        lamda = lamda+0.01
    plt.plot(x, y_origin,'r',marker='o',label='y=sin(2*pi*x)')
    plt.legend()
    plt.show()
draw()
# draw100()