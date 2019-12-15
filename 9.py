from functools import reduce
#S:-1,M:0,L:1
data=[[1,-1,-1],[1,0,-1],[1,0,1],[1,-1,1],[1,-1,-1],[2,-1,-1],[2,0,-1],[2,0,1],[2,1,1],[2,1,1],[3,1,1]]
alpha=1
def prior(data):
    y=list(map(lambda x:abs(x[-1]+1)/2,data))
    y1=reduce(lambda x,y:x+y,y)/len(y)
    index=[[],[]]
    for i in range(len(y)):
        if y[i]==0:
            index[0].append(i)
        else:
            index[1].append(i)
    return [1-y1,y1],index
def cal(x,xl,l):
    num=0
    for i in l:
        if xl[i] == x:
            num = num + 1
    return (num+alpha) /(len(l)+alpha*3)
def likelihood(data,x1,x2):
    x_1=list(map(lambda x:x[0],data))
    x_2=list(map(lambda x:x[1],data))
    y=list(map(lambda x:x[-1],data))
    prior_y,index=prior(data)
    p_x1_y0=cal(x1,x_1,index[0])
    p_x1_y1=cal(x1,x_1,index[1])
    p_x2_y0=cal(x2,x_2,index[0])
    p_x2_y1=cal(x2,x_2,index[1])

    p_y0=p_x1_y0*p_x2_y0*prior_y[0]

    p_y1=p_x1_y1 * p_x2_y1 * prior_y[1]

    return p_y0,p_y1

if __name__ == '__main__':
    p_y0,p_y1=likelihood(data,1,0)
    print("Laplace smooth alpha:",alpha)
    print("class -1:",p_y0)
    print("class  1:",p_y1)
    print("归一化后概率：")
    print("class -1:", p_y0/(p_y0+p_y1))
    print("class  1:", p_y1/(p_y0+p_y1))

