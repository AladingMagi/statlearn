import numpy as np
import cv2
from sklearn.svm import LinearSVC
from functools import reduce
from sklearn.decomposition import PCA
import shutil
import joblib
import os
import time
def pHashValue(img):
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    img = cv2.dct(img)
    img = img[:8, :8]
    avg = np.sum(img)/64*np.ones((8, 8))
    temp = img-avg
    temp[temp >= 0] = 1
    temp[temp < 0] = 0
    temp = temp.reshape((1,64))
    return temp


def pHash(img1, img2):
    img1 = pHashValue(img1)
    img2 = pHashValue(img2)
    result = np.nonzero(img1-img2)
    result = np.shape(result[0])[0]
    return result

def getsSimilarSave(k,img,path):
    simi_list=[]
    for i in range(50):
        temp=path + "\\image" + str(i) + ".jpg"
        img2=cv2.imread(temp)
        img2=cv2.resize(img2, (300, 200), interpolation=cv2.INTER_AREA)
        simi_list.append(pHash(img,img2))

    if k <0 or k>50:
        print("Sorry,each class in our lib only have 50 image!")
        return

    shutil.rmtree(".\\res\\")
    os.mkdir(".\\res")
    for i in range(k):
        min_i=simi_list.index(min(simi_list))
        simi_list.remove(min(simi_list))
        src=path+"\\image"+str(min_i)+".jpg"
        dest=".\\res\\similar"+str(i)+".jpg"
        shutil.copyfile(src,dest)

def getfeat(img):
    winSize = (128,128)
    blockSize = (64,64)
    blockStride = (8,8)
    cellSize = (16,16)
    nbins = 9
    # cv2.imshow("t",img)
    # cv2.waitKey(0)
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    winStride = (32,32)
    padding = (8,8)
    hog_res = hog.compute(img, winStride, padding).reshape((-1,))
    return hog_res

def svm(feat,label):
    clf = LinearSVC(max_iter=10000)
    clf.fit(feat, label)
    return clf

def predict(clf,feat):
    return clf.predict(feat)

def imgPreprocess(path):
    temp=path + "\\image" + str(0) + ".jpg"
    img = cv2.imread(temp)
    img = cv2.resize(img, (300, 200), interpolation=cv2.INTER_AREA)
    cur=getfeat(img)
    for i in range(1,50):
        temp=path+"\\image"+str(i)+".jpg"
        img=cv2.imread(temp)
        img = cv2.resize(img, (300,200) ,interpolation=cv2.INTER_AREA)
        new=getfeat(img)
        cur=np.vstack((cur,new))
    return cur

def pcaProcess(train,n):
    pca = PCA(n_components=n)
    newdata = pca.fit_transform(train)
    print("PCA保留维数：",n)
    print("保留维数主成分占比：",np.sum(pca.explained_variance_ratio_))
    return newdata,pca

if __name__ == '__main__':
    time_start = time.time()
    clazz = ['accordion', 'airplanes', 'bass', 'bonsai', 'brain', 'buddha',
             'butterfly', 'camera', 'car_side', 'cellphone', 'chair', 'chandelier',
             'clutter', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head',
             'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar',
             'elephant', 'emu', 'euphonium', 'ewer', 'Faces', 'ferry', 'flamingo',
             'gramophone', 'grand_piano', 'hawksbill', 'hedgehog', 'helicopter', 'ibis',
             'joshua_tree', 'kangaroo', 'ketch', 'lamp']
    testClazz = ['accordion', 'airplanes', 'bass', 'bonsai', 'brain', 'buddha',
             'butterfly', 'camera', 'car_side', 'cellphone', 'chair', 'chandelier',
             'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head',
             'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar',
             'elephant', 'emu', 'euphonium', 'ewer', 'Faces', 'ferry', 'flamingo',
             'gramophone', 'grand_piano', 'hawksbill', 'hedgehog', 'helicopter', 'ibis',
             'joshua_tree', 'kangaroo', 'ketch', 'lamp']
    # feat=[]
    # label=[]
    # for i in range(len(testClazz)):
    #     path=".\\imgTrain\\"+testClazz[i]
    #     res=imgPreprocess(path)
    #     feat.append(res)
    #     for j in range(50):
    #         label.append(i)
    # feat=reduce(lambda x,y:np.vstack((x,y)),feat)
    # label=np.array(label)
    # print("图像原特征维数：",len(feat[0]))
    #
    # train1,pca=pcaProcess(feat,1000)

    # np.savetxt(".\\feature.txt", train, fmt='%f', delimiter=',')
    # np.savetxt(".\\label.txt",label, fmt='%f', delimiter=',')
    # joblib.dump(pca,'PCA2')
    # train = np.loadtxt('.\\feature.txt',delimiter=',')
    # label1 = np.loadtxt('.\\label.txt',delimiter=',' )
    # clf=svm(train, label1)
    # joblib.dump(clf, 'SVM2')

    # pca = joblib.load('.\\PCA2')
    # clf=joblib.load('.\\SVM2')
    # k=10
    # res_list = []
    # index=[]
    # pr_list=[]
    # re_list=[]
    # f1_list=[]
    # mrr_list=[]
    # for c in range(40):
    #     print("-------------------------------------------")
    #     print("当前测试类别为:", testClazz[c])
    #     res_list.clear()
    #     sum=0
    #     index.clear()
    #     for i in range(10):
    #         path2=".\\imgTest\\"+testClazz[c]+"\\image"+str(i)+".jpg"
    #         img = cv2.imread(path2)
    #         img = cv2.resize(img, (300, 200), interpolation=cv2.INTER_AREA)
    #         f=getfeat(img)
    #         f=pca.transform(f.reshape(1,-1))
    #         r = predict(clf, f)
    #         res=testClazz[int(r)]
    #         res_list.append(res)
    #     print("预测结果为：")
    #     print(res_list)
    #     for n in res_list:
    #         if n== testClazz[c]:
    #             sum=sum+1
    #             index.append(res_list.index(n)+1)
    #     pr=sum/10
    #     re=sum/50
    #     f1=2*pr*re/(pr+re)
    #     mrr=reduce(lambda x,y:1/x+1/y,index)/len(index)
    #     pr_list.append(pr)
    #     re_list.append(re)
    #     f1_list.append(f1)
    #     mrr_list.append(mrr)
    #     print("精确率：",pr)
    #     print("召回率：",re)
    #     print("F1-value:",f1)
    #     print("MRR:",mrr)
    # print("-------------------------------------------------------")
    # print("总平均精确率：", reduce(lambda x,y:x+y,pr_list)/len(pr_list))
    # print("总平均召回率：", reduce(lambda x,y:x+y,re_list)/len(re_list))
    # print("总平均F1-value:", reduce(lambda x,y:x+y,f1_list)/len(f1_list))
    # print("总平均MRR:", reduce(lambda x,y:x+y,mrr_list)/len(mrr_list))
    # time_end = time.time()
    # print("测试耗时：",time_end-time_start,"sec")

    pca = joblib.load('.\\PCA2')
    clf = joblib.load('.\\SVM2')
    k = 10
    input_img_path2=".\\imgTest\\"+testClazz[19]+"\\image"+str(0)+".jpg"
    img = cv2.imread(input_img_path2)
    img = cv2.resize(img, (300, 200), interpolation=cv2.INTER_AREA)
    f=getfeat(img)
    f=pca.transform(f.reshape(1,-1))
    r = predict(clf, f)
    res=testClazz[int(r)]
    print(res)
    res_path = ".\\imgTrain\\" + res
    getsSimilarSave(k, img, res_path)
    time_end = time.time()
    print("检索到",k,"张图片，保存在res目录下")
    print("检索耗时：",time_end-time_start,"sec")