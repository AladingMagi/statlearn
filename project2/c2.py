from sklearn import neighbors          
from c1 import x,y
import numpy as np 

print(x.shape)
def build_knn(C=4,k=2,weight='uniform',M='manhattan'): 

    knn = neighbors.KNeighborsClassifier(n_neighbors=k,weights=weight,metric=M)
    if C != 4:
        xr = np.concatenate([x[:C*36],x[C*36+36:]])
        yr = np.concatenate([y[:C*36],y[C*36+36:]])
        knn.fit(xr,yr)
        xl = x[C*36:C*36+36]
        yl = y[C*36:C*36+36]
        accuracy = knn.score(xl,yl)
    else :
        xr = x[:C*36]
        yr = y[:C*36]
        knn.fit(xr,yr)
        xl = x[C*36:]
        yl = y[C*36:]
        accuracy = knn.score(xl,yl)
    return accuracy

if __name__ == "__main__":
    sm = 0
    for i in range(5):
        accuracy = build_knn(C=i)
        sm += accuracy
    avge = sm / 5
    print(avge)


    
    


# knn.fit(x,y)                 #用KNN的分类器进行建模，这里利用的默认的参数，大家可以自行查阅文档

