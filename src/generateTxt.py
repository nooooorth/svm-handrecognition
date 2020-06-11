import os
import cv2
import numpy as np

# dataPath：图片源文件夹
# storePath: 输出文件夹
# label：标签值
# 输出：为每个文件夹中的所有文件名生成一个txt文件，以label.txt命名
def generateTxt(dataPath, storePath, label):
    if not os.path.exists(storePath):
        os.mkdir(storePath)
    listTxt = open(storePath + "/" + str(label) + ".txt",'w')
    for filename in os.listdir(dataPath+ "/" +str(label)):
        # print(filename)
        if filename == "../data/txtList/list.txt":
            continue
        listTxt.write(dataPath+ "/" + str(label) + "/" + filename + " " + str(label) + "\n")
    listTxt.close()

# storePath:txt存放文件夹
# 输出：合并所有txt文件中的内容
def mergeTxt(storePath):
    listMerge = open(storePath + "/list.txt", 'w')
    for fileName in os.listdir(storePath):
        txtFiles = open(storePath + '/' + fileName, 'r')
        if txtFiles == "../data/txtList/list.txt":
            continue
        content = txtFiles.read()       # 使用readline()仅读取第一行
        for item in content:
            listMerge.write(item)
    listMerge.close()

# 输入：图片文件夹路径，txt存放路径，label
# 输出：输出（28*28）一维numpy数组 + label
def generateMatrix(dataPath, storePath , i):
    print("strat!")
    data = np.empty(len(os.listdir()),785)
    for fileName in os.listdir(dataPath+ "/" +str(i)):
        # print("当前处理文件:",fileName)
        img = cv2.imread(dataPath + "/" + fileName, 0)
        img = cv2.resize(img,(28,28))
        img = img.flatten()
        data[i] = img           # 前面的值为
        data[i][784] = i        # 最后一列=label
    print("done!")

if __name__ == "__main__":
    dataPath ="../data/crHand"
    storePath = "../data/txtList"
    i = 0
    while(i <= 9):
        # generateTxt(dataPath, storePath, i)
        generateMatrix(dataPath, storePath, i)
        i = i + 1
    # mergeTxt(storePath)