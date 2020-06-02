import numpy as np
from sklearn import svm
from sklearn.model_selection import  train_test_split
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score,fbeta_score
from sklearn.metrics import precision_recall_fscore_support, classification_report

# 标签转换
def label_type(s):
    it = {"finger":0,"palm":1,"other":2}
    return it[str(s,encoding="utf-8")]

def svmTrain(dataPath,startFeatureCol, endFeatureCol, labelCol, modelPath):
    data = np.loadtxt(dataPath,dtype=float,delimiter=' ',converters={labelCol: label_type})
    x, y = np.split(data, (labelCol,), axis=1)
    x = x[:, startFeatureCol:endFeatureCol]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size= 0.90)

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


if __name__ == "__main__":
    dataPath = "data.txt"       # 训练数据路径
    startFeatureCol = 0         # 数据开始列
    endFeatureCol = 2           # 数据结束列
    labelCol=2                  # 标签列
    modelPath = "test.m"        # 模型存放路径
    svmTrain(dataPath, startFeatureCol, endFeatureCol, labelCol, modelPath)
    x = 0        # 预测值
    print(svmPredict(x,modelPath))