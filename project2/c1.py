import re
import linecache
import numpy as np
import os
 
def get_Tdata(file='wine.txt'):
    with open(file,encoding="utf-8") as f: 
        f.seek(0) #把指针移到文件开头位置
        y = []
        x = []
        for line in f.readlines():#readlines以列表输出文件内容
            line=line.replace("\n","")#改变元素，去掉，和换行符\n,tab键则把逗号换成"/t",空格换成" "
            line = line.split(",")
            y.append(float(line[0]))
            k = []
            for i in range(1,len(line)):
                k.append(float(line[i]))
            x.append(k)
    return y,x

y,x = get_Tdata()
y = np.array(y)
x = np.array(x)

arr = np.arange(178)
np.random.shuffle(arr)
for i in range(len(y)):
    y[i] = y[arr[i]]
    x[i] = x[arr[i]]

