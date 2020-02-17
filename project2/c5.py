from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import neighbors 
import numpy as np 
from c1 import x,y



def build_knn(C=4,k=2,weight='uniform',M='manhattan'): 

    knn = neighbors.KNeighborsClassifier(n_neighbors=k,weights=weight,metric=M)
    if C != 4:
        xr = np.concatenate([x[:C*36],x[C*36+36:]])
        yr = np.concatenate([y[:C*36],y[C*36+36:]])
        xl = x[C*36:C*36+36]
        yl = y[C*36:C*36+36]
        X_train_dunction = SelectKBest(chi2,k=3).fit_transform(xr,yr)
        X_test_dunction = SelectKBest(chi2,k=3).fit_transform(xl,yl)
        knn.fit(X_train_dunction, yr)
        accuracy = knn.score(X_test_dunction,yl)
    else :
        xr = x[:C*36]
        yr = y[:C*36]
        xl = x[C*36:]
        yl = y[C*36:]
        X_train_dunction = SelectKBest(chi2,k=3).fit_transform(xr,yr)
        X_test_dunction = SelectKBest(chi2,k=3).fit_transform(xl,yl)
        knn.fit(X_train_dunction, yr)
        accuracy = knn.score(X_test_dunction,yl)
        # print(pca.explained_variance_ratio_)
    return accuracy

if __name__ == "__main__":
    sm = 0
    for i in range(5):
        accuracy = build_knn(C=i)
        sm += accuracy
    avge = sm / 5
    print(avge)
