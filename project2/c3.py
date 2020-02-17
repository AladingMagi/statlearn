from sklearn import neighbors 
from sklearn.decomposition import PCA
import numpy as np 
from c1 import x,y



def build_knn(C=4,k=2,weight='uniform',M='manhattan'): 

    knn = neighbors.KNeighborsClassifier(n_neighbors=k,weights=weight,metric=M)
    if C != 4:
        xr = np.concatenate([x[:C*36],x[C*36+36:]])
        yr = np.concatenate([y[:C*36],y[C*36+36:]])
        pca = PCA(n_components=5)
        pca.fit(xr,yr)
        xl = x[C*36:C*36+36]
        yl = y[C*36:C*36+36]
        X_train_dunction = pca.transform(xr)
        X_test_dunction = pca.transform(xl)
        knn.fit(X_train_dunction, yr)
        accuracy = knn.score(X_test_dunction,yl)
        print(pca.explained_variance_ratio_)
        ##pca.n_components_
    else :
        xr = x[:C*36]
        yr = y[:C*36]
        pca = PCA(n_components=5)
        pca.fit(xr,yr)
        xl = x[C*36:]
        yl = y[C*36:]
        X_train_dunction = pca.transform(xr)
        X_test_dunction = pca.transform(xl)
        knn.fit(X_train_dunction, yr)
        accuracy = knn.score(X_test_dunction,yl)
        print(pca.explained_variance_ratio_)
    return accuracy

if __name__ == "__main__":
    sm = 0
    for i in range(5):
        accuracy = build_knn(C=i)
        sm += accuracy
    avge = sm / 5
    print(avge)
