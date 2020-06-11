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
    clf = svm.SVC(C=0.8, kernel="rbf", gamma=20, decision_function_shape="ovr")
    clf.fit(x_train, y_train.ravel())
    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))
    joblib.dump(clf, modelPath)
    return clf


def svmPredict(x, modelPath):
    clf = joblib.load(modelPath)
    resultPredict = clf.predict(x)
    return resultPredict

# 图片转一维数组，x的格式
def imgToX(img):
    img_array = np.empty((0,783))
    img = cv2.resize(img,(28,28))
    img = img.flatten()
    img_array[0] = img
    return img_array

if __name__ == "__main__":
    dataPath = "../data/txtList/date.csv"     # 数据文件路径
    modelPath = "../data/model/hand.m"  # 模型存放路径
    labelCol= 784                   # 标签列
    svmTrain(dataPath, labelCol, modelPath)

    # predict
    x = imgToX(cv2.imread("../data/crHand/1/1_20.jpg",0))        # 预测值
    print(svmPredict(x,modelPath))