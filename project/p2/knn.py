import numpy as np
# import matplotlib.pyplot as plt
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

def pre(x):
    x=x[:-1].split(",")
    x=map(lambda i:float(i),x)
    return list(x)
def readData():
    data_path='C:\\Users\\Administrator\\Desktop\\data2\\wine.data'
    f = open(data_path,"r")
    data = f.readlines()
    f.close()
    data=list(map(pre,data))
    random.shuffle(data)
    label=list(map(lambda x:x[0],data))
    train=list(map(lambda x:x[1:14],data))
    return np.array(label),np.array(train)

def simple_knn(k,train,label):
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto', weights='uniform', metric='euclidean')
    knn.fit(train, label)
    answer = knn.predict(train)
    print("k:",k)
    print("simple_knn accuracy:", np.mean(answer == label))
    return answer

def kfold(train,label):
    train_X = None
    train_y = None
    test_X = None
    test_y = None
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(train):
        train_X, train_y = train[train_index], label[train_index]
        test_X, test_y = train[test_index], label[test_index]
    return train_X, train_y,test_X, test_y

def crossValid(train,label):
    k_range = range(1, 100)
    k_error = []
    accuracy = []
    train_X, train_y, test_X, test_y=kfold(train,label)
    for k in k_range:
        # knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto', weights='uniform', metric='euclidean')
        knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto', weights='distance', metric='manhattan')
        knn.fit(train_X, train_y)
        res = knn.score(test_X,test_y)
        accuracy.append(res)
    print("k值1-100对应精确率：")
    print(np.array(accuracy))
    return accuracy.index(max(accuracy))+1,max(accuracy)
def pcaProcess(train,n):
    pca = PCA(n_components=n)
    newdata = pca.fit_transform(train)
    print("保留维数：",n)
    print("各成分占比：",pca.explained_variance_ratio_)
    return newdata
def ldaProcess(train,label,n):
    lda = LinearDiscriminantAnalysis(n_components=n)
    lda.fit(train,label)
    newdata = lda.transform(train)
    print("LDA保留维数：", n)
    return newdata
def featureSelect(train,label):
    lasso = Lasso()
    lasso.fit(train, label)
    model = SelectFromModel(lasso,prefit=True)
    newdata = model.transform(train)
    return newdata
if __name__ == "__main__":
    label, train = readData()
    # simple_knn(4,train,label)
    # print("Use weights='distance', metric='manhattan':")
    # k=crossValid(train, label)
    # print("The best k is:",k)
    # train1 = featureSelect(train,label)

    # for i in range(1,14):
    #     train1= pcaProcess(train,14-i)
    #     accuracy=crossValid(train1,label)
    #     print("accuracy:",accuracy)
    #     print("----------------------------------------------")

    # for i in range(1,14):
    #     train1= ldaProcess(train,label,i)
    #     accuracy=crossValid(train1,label)
    #     print("accuracy:",accuracy)
    #     print("----------------------------------------------")
    print("Feature selection based on Lasso:")
    train1=featureSelect(train, label)
    k = crossValid(train1, label)
    print("The best k is:", k)