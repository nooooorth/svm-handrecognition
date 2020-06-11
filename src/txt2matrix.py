import numpy as np
import cv2
import os

def txtToMatrix(srcPath,matrixPath,labelPath):
    print("start！")
    data = np.loadtxt(srcPath, dtype=str, delimiter= " ")
    srcPicPath , label = np.split(data,(1,),axis=1)
    i = 0
    picArray = np.empty((len(srcPicPath),784))
    labelArray = np.empty((len(label),1))
    while(i < len(srcPicPath)):
        # print("当前处理：",srcPicPath[i][0])
        img = cv2.imread(srcPicPath[i][0],0)
        img = cv2.resize(img, (28,28))
        img = img.flatten()
        picArray[i] = img
        labelArray[i] = label[i][0]
        i = i + 1
    np.savetxt(matrixPath, picArray, fmt="%d", delimiter=",", newline="\n")
    np.savetxt(labelPath, labelArray ,fmt="%d", delimiter=",", newline="\n")
    print("Done!")


if __name__ == "__main__":
    srcPath = "../data/txtList/list.txt"
    matrixPath = "../data/txtList/data.csv"
    labelPath = "../data/txtList/label.csv"
    txtToMatrix(srcPath,matrixPath,labelPath)