import numpy as np
import matplotlib.pyplot as plt
data = [[1,2,1],[2,3,1],[3,3,1],[2,1,-1],[3,2,-1]]
eta =1

def sign(val):
    return -1 if val < 0 else 1

def perceptron(data_p):
    w = np.array([0,0])
    b = 0
    flag = True
    while(flag):
        flag = False
        for x in data_p:
            tmpx = np.array([x[0],x[1]])
            if x[-1]*(w.dot(tmpx.T)+b)<= 0:
                w = w + eta*x[-1]*tmpx
                b = b + eta*x[-1]
                flag = True
    return w,b

def test(data_t,w,b):
     for x in data_t:
         tmpx = np.array([x[0],x[1]])
         y = sign(w.dot(tmpx.T)+b)
         print("input:",tmpx,"output:",y)

def draw(data_t,w,b):
    for x in data_t:
        if x[-1] == 1:
            plt.plot(x[0], x[1], 'b', marker='o')
        else:
            plt.plot(x[0], x[1], 'y', marker='*')
    x0 = np.linspace(0, 5, 500)
    x1 = -(w[0]*x0 + b)/w[1]
    plt.plot(x0, x1, 'r', marker='.')
    plt.show()
if __name__ == "__main__":
    w,b=perceptron(data)
    test(data,w,b)
    draw(data,w,b)