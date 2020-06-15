import cv2
import numpy as np
import os

# Ycrcb空间肤色分割，自动阈值分割，效果好
# Img，输入BGR图像
# YCrMinThreshold, YCbMinThreshold, YCrMaxThreshold, YCbMaxThreshold，Cr和Cb通道肤色阈值下限和上限
# 返回值：
# ycrcbFrame, YCrCb图像
# ycrFrame, Cr通道图像
# ycrcbHand，肤色区域掩膜
def ycrcbSeg(Img, YCrMinThreshold=135, YCrMaxThreshold=175, YCbMinThreshold=80, YCbMaxThreshold=160):
    ycrcbFrame = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)  # 转换到YCrCb空间
    ycrcbFrame = cv2.GaussianBlur(ycrcbFrame, (5, 5), 0)  # 高斯滤波
    ycrFrame = ycrcbFrame[:, :, 1]  # 转换到YCrCb空间后取Cr通道
    ycrFrame[ycrFrame > YCrMaxThreshold] = 0  # 抑制非肤色像素
    ycrFrame[ycrFrame < YCrMinThreshold] = 0
    thres, ycrcbHand = cv2.threshold(ycrFrame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(thres)
    if thres < 130:  # 当自动分割阈值thres低于130时，会把背景分割进来，此刻采用固定阈值分割
        ycrcbHand = cv2.inRange(ycrcbFrame, np.array([0, YCrMinThreshold, YCbMinThreshold]),
                                np.array([255, YCrMaxThreshold, YCbMaxThreshold]))  # 肤色分割
    return ycrcbFrame, ycrFrame, ycrcbHand

def handShape(ycrcbHand):
    _, handContour, _ = cv2.findContours(ycrcbHand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    handContour = handContour[0]

    handArea = cv2.contourArea(handContour)  # 手部面积
    # 手部最小外接矩形，返回一个Box2D结构rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）
    rect = cv2.minAreaRect(handContour)
    w, h, wIdx, hIdx = cv2.minMaxLoc(rect[1])  # 外接矩形短边长、长边长、短边索引号、长边索引号
    rectArea = w * h  # 手部最小外界矩形面积

    handAreaRatio = handArea / rectArea  # 手部面积占比
    handAreaRatio_100 = handAreaRatio * 100  # 手部面积百分比
    # 长宽比，确保AspectRatio>1
    if (h > w):
        AspectRatio = h / w
    elif (w > h):
        AspectRatio = w / h
    predictValue = [[handAreaRatio_100, AspectRatio]]
    return predictValue

if __name__ == "__main__":
    handDataPath = "F:/Project/svm-handrecognition/data/Dataset/9"     # 源图片路径
    crHandPath = "F:/Project/svm-handrecognition/data/crHand/9"        # 存放文件夹路径
    pic_number = 0      # 文件重命名起始

    '''
    ## 将原始图像转换为cr图像储存
    for fileName in os.listdir(handDataPath):
        print("读取文件名为：",fileName)
        if(fileName==".DS_Store"):
            continue
        img = cv2.imread(handDataPath + "/" + fileName)
        _, crPic, ycrcbHand = ycrcbSeg(img)
        cv2.imwrite(crHandPath +'/' + '9_' + str(pic_number) + '.jpg', ycrcbHand)
        pic_number = pic_number + 1
    '''
    img = cv2.imread("../data/Dataset/1/IMG_1119.JPG")
    _, crPic, ycrcbHand = ycrcbSeg(img)
    handShape(img, ycrcbHand)