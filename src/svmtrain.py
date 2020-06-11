import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import  train_test_split
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score,fbeta_score
from sklearn.metrics import precision_recall_fscore_support, classification_report

# SVM训练函数
# dataPath:数据文件路径（txt/csv）
# labelCol:标签列
# modelPath:模型存放路径
def svmTrain(dataPath, labelCol, modelPath):
    # 训练数据处理
    data = np.loadtxt(dataPath, dtype=int, delimiter=",")
    x, y = np.split(data, (labelCol,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)   # 训练/测试比7:3

    # SVM参数设置及训练
    # svm.SVC(C=1.0,        C-SVC惩罚松弛变量，越大表示不能容忍误差，泛化能力差
    # kernel='rbf',         kernal:核函数 rbf,linear,poly,sigmoid,precomputed
    # degree=3,             degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略
    # gamma='auto',         gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/样本量
    # coef0=0.0,            coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
    # tol=0.001,            tol ：停止训练的误差值大小，默认为1e-3
    # max_iter=-1,          max_iter ：最大迭代次数。-1为无限制。
    # decision_function_shape=None      decision_function_shape ：‘ovo’, ‘ovr’  ovo-1V1,ovr-1V多
    # )
    clf = svm.SVC(C=0.8, kernel="rbf", gamma=20, decision_function_shape="ovr")
    clf.fit(x_train, y_train)
    print("训练集：",clf.score(x_train, y_train))
    print("测试集：",clf.score(x_test, y_test))
    joblib.dump(clf, modelPath)
    # return clf


def svmPredict(x, modelPath):
    clf = joblib.load(modelPath)
    resultPredict = clf.predict(x)
    return resultPredict

# 图片转一维数组，x的格式
def imgToX(img):
    img_array = np.empty((0,783))
    img = cv2.resize(img,(28,28))
    img = img.flatten()
    img_array = img.reshape(-1,784)
    return img_array

if __name__ == "__main__":
    # dataPath = "../data/txtList/date.csv"     # 数据文件路径
    modelPath = "../data/model/hand.m"  # 模型存放路径
    # labelCol= 784                   # 标签列
    # svmTrain(dataPath, labelCol, modelPath)

    # predict
    x = imgToX(cv2.imread("../data/crHand/7/7_15.jpg",0))        # 预测值
    print("Predict:",svmPredict(x,modelPath))