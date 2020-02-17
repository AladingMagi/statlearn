# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:31:57 2019
@author: wuzhe
"""
import cv2
from math import floor
import numpy as np
# import dhash
# from PIL import Image

def HashValue(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = floor(img[i,j]/4)
    avg = np.sum(img)/64*np.ones((8, 8))
    temp = img-avg
    temp[temp >= 0] = 1
    temp[temp < 0] = 0
    temp = temp.reshape((1,64))
    return temp

def Hash(img1, img2):
    img1 = HashValue(img1)
    img2 = HashValue(img2)
    result = np.nonzero(img1-img2)
    result = np.shape(result[0])[0]
    if result<=5:
        print('Same Picture')
    return result

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
    if result<=5:
        print('Same Picture')
    return result


def DHashValue(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    img2 = []
    for i in range(8):
        img2.append(np.array(img[:,i])-np.array(img[:,i+1]))
    img2 = np.mat(img2).T
    img2[img2 >= 0] = 1
    img2[img2 < 0] = 0
    img2 = img2.reshape((1,64))
    return img2


def DHash(img1, img2):
    img1 = DHashValue(img1)
    img2 = DHashValue(img2)
    result = np.nonzero(img1-img2)
    result = np.shape(result[0])[0]
    if result<=5:
        print('Same Picture')
    return result

#
def dHash_use_package(img1, img2):
    image1 = Image.open(img1)
    image2 = Image.open(img2)
    row1, col1 = dhash.dhash_row_col(image1)
    row2, col2 = dhash.dhash_row_col(image2)
    a1 = int(dhash.format_hex(row1, col1), 16)
    a2 = int(dhash.format_hex(row2, col2), 16)
    result = dhash.get_num_bits_different(a1, a2)
    if result<=5:
        print('Same Picture')
    return result
def readImage(name):
    path = ".\\data3\\"+name
    img = cv2.imread(path)
    return img

if __name__ == "__main__":
    img1=readImage("butterfly\\image_0001.jpg")
    img2=readImage("airplanes\\image_0002.jpg")
    print(Hash(img1, img2))
    print(pHash(img1, img2))
    print(DHash(img1, img2))