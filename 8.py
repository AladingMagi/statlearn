import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

x = np.array([[1,-1],[1,0],[1,0],[1,-1],[1,-1],[2,-1],[2,0],[2,0],[2,1],[2,1],[3,1]])
y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1])
plt.scatter(x[:, 0], x[:, 1], marker='^', c=y)
plt.show()
bdt = AdaBoostClassifier(DecisionTreeClassifier(),
                         algorithm="SAMME",learning_rate=0.1)
bdt.fit(x, y)


x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(x[:, 0], x[:, 1], marker='^', c=y)
plt.show()
