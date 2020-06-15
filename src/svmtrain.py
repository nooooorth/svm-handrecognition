import numpy as np
import cv2
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import  train_test_split,GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score,fbeta_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from handSeg import ycrcbSeg
from picHu import imgToHu

# SVM训练函数
# dataPath:数据文件路径（txt/csv）
# labelCol:标签列
# modelPath:模型存放路径
def svmTrain(dataPath, labelCol, modelPath):
    # 训练数据处理
    data = np.loadtxt(dataPath,dtype=int,delimiter=",")
    data = data[~np.isnan(data).any(axis=1)]        # 删除含有缺省值的行
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

def svmTrain_Grid(dataPath, labelCol, modelPath):
    data = np.loadtxt(dataPath, dtype=int, delimiter=",")
    data = data[~np.isnan(data).any(axis=1)]  # 删除含有缺省值的行
    x, y = np.split(data, (labelCol,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)  # 训练/测试比7:3

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        # 用训练集训练这个学习器 clf
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()

        # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
        print("最好的参数搭配结果:",clf.best_params_)

        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        # 看一下具体的参数间不同数值的组合后得到的分数是多少
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(x_test)

        # 打印在测试集上的预测结果与真实值的分数
        print(classification_report(y_true, y_pred))

        print()

    joblib.dump(clf, modelPath)

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
    dataPath = "../data/txtList/date.csv"     # 数据文件路径
    modelPath = "../data/model/hand_1.m"  # 模型存放路径
    labelCol= 784                   # 标签列
    # svmTrain(dataPath, labelCol, modelPath)
    svmTrain_Grid(dataPath, labelCol, modelPath)
    # predict
    img = cv2.imread("../data/Dataset/9/IMG_1137.JPG")
    _,_,img  = ycrcbSeg(img)
    x = imgToX(img)        # 预测值
    print("Predict:",svmPredict(x,modelPath))