import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
data = [[1,2,1],[2,3,1],[3,3,1],[2,1,-1],[3,2,-1]]
x=np.array([[1,2],[2,3],[3,3],[2,1],[3,2]])
y=np.array([1,1,1,-1,-1])
svc = svm.SVC(kernel='linear')
svc.fit(x, y)
w=svc.coef_[0]
k=-w[0]/w[1]
x1=np.linspace(0,6)
y1=k*x1-(svc.intercept_[0])/w[1]
for x in data:
    if x[-1] == 1:
        plt.plot(x[0], x[1], 'b', marker='o')
    else:
        plt.plot(x[0], x[1], 'y', marker='*')
res = svc.predict( [[2,5],[1,1],[0,3],[2,2],[1,0]])
i=0
for x in [[2,5],[1,1],[0,3],[2,2],[1,0]]:
    if res[i] == 1:
      plt.plot(x[0], x[1], 'r', marker='^')
    else:
      plt.plot(x[0], x[1], 'g', marker='^')
    i=i+1
plt.plot(x1,y1)
plt.show()
res = svc.predict( [[2,5],[1,1],[0,3],[2,2],[1,0]])
print( res )
