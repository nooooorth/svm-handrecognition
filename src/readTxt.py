import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import  train_test_split
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score,fbeta_score
from sklearn.metrics import precision_recall_fscore_support, classification_report

# 输入图片txt文件：文件名 标签
# 返回文件名列表
def readTxt(path):
    data = []
    for line in open(path):
        line = line[:-3]
        data.append(line)
    # print(data[0])
    return data

# 输入图片路径list
def showImg(dataList):
    for item in dataList:
        img = cv2.imread(item)
        cv2.imshow("test",img)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    path = "../data/txtList/date.csv"     # 数据文件路径
    data = np.loadtxt(path, dtype= int , delimiter=",")
    # print(data)
    # 最后一列（784列为标签）
    x, y = np.split(data, (784,), axis= 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
    print("done")

