import os
import cv2
import numpy as np

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
    path="../data/txtList/list.txt"
    dataList = readTxt(path)    # 读取txt中文件名保存至变量
    # showImg(dataList)       # 显示图片