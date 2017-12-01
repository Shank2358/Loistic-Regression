# -*- coding: utf-8 -*-
from numpy import *
from os import listdir

#数据载入
def loadData(direction):
    print(direction)
    dataArray = []#变量数据
    labelArray = []#类别标签
    trainfileList = listdir(direction)#文件名
    m = len(trainfileList)#文件数
    for i in range(m):
      filename = trainfileList[i]
      fr = open('%s/%s' % (direction, filename))
      for line in fr.readlines():
          lineArr = line.strip().split( )
          dataArray.append([float(lineArr[0]),float(lineArr[1]),float(lineArr[3]),float(lineArr[4]),float(lineArr[5]),float(lineArr[6]),float(lineArr[7])])
          labelArray.append([int(lineArr[8])])
      fr.close()
    # data=transpose(dataArray)
    # label = transpose(labelArray)
    # print(data)
    # print(label)
    return dataArray, labelArray

# sigmoid(inX)函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

#梯度下降法
def gradAscent(dataArray, labelArray, alpha, maxCycles):
    dataMat = mat(dataArray)  # size:m*n
    labelMat = mat(labelArray)  # size:m*1
    m, n = shape(dataMat)
    weigh = mat(ones((n, 1)))
    for i in range(maxCycles):
        h = sigmoid(dataMat * weigh)
        error = labelMat - h  # size:m*1
        weigh = weigh + alpha * dataMat.transpose() * error
    return weigh


# 分类函数，根据参数weigh对测试样本进行预测，同时计算错误率
def classfy(testdir, weigh):
    dataArray, labelArray = loadData(testdir)
    dataMat = mat(dataArray)
    labelMat = mat(labelArray)
    h = sigmoid(dataMat * weigh)  # size:m*1，logistic函数值（0-1）
    print(h)
    m = len(h)
    error = 0.0
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0
    R=mat(ones([m,1]))#分类情况,TP=1，FP=0,TN=3,FN=2
    for i in range(m):
        if float(h[i]) > 0.5:
            print(int(labelMat[i]), 'is classfied as: 1')
            if int(labelMat[i]) != 1:
                error += 1
                print('error')
                R[i]=0
                FP += 1
            else:
                R[i]=1
                TP += 1
        else:
            print(int(labelMat[i]), 'is classfied as: 0')
            if int(labelMat[i]) != 0:
                error += 1
                print('error')
                R[i]=2
                FN += 1
            else:
                R[i] =3
                TN += 1
    print(R)
    Error_rate=error / m
    Accuracy=1-Error_rate
    Precision=TP / (TP+FP)
    Recall=TP / (TP + FN)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    Sensitivity=TP/(TP+FN)#灵敏性即欺诈交易识别率，指的是被预测正确的欺诈交易数量占总欺诈交易数量的比例，其反映了风险识别模型对欺诈交易的识别情况
    Specificity=TN/(TN+FP)#特异性是指被预测正确的正常交易数量占正常交易总数的比例。
    print('Error rate is:', '%.4f' % (error / m))
    print('Accuracy is:', '%.4f' % Accuracy)
    print('Precision is:', '%.4f' % (TP /(TP+FP) ))#查准率，真正例占所有预测正例的比例，（检测出的正例中有多少是真正的正例）
    print('Recall is:', '%.4f' % (TP / (TP + FN)))#查全率，预测出的真正例占所有实际正例的比例，（真正的正例有多少被检测出来）
    print('F1 is:', '%.4f' % F1)
    print('Sensitivity is:', '%.4f' % Sensitivity)
    print('Specificity is:', '%.4f' % Specificity)
    return R,h,error


"""
用loadData函数从train里面读取训练数据，接着根据这些数据，用gradAscent函数得出参数weigh，最后就可以用拟
合参数weigh来分类了。
"""
#mian函数
def digitRecognition(trainDir, testDir, alpha, maxCycles):
    data, label = loadData(trainDir)
    weigh = gradAscent(data, label, alpha, maxCycles)
    R,h,error=classfy(testDir, weigh)
    return R,h,error

# 运行函数
R,h,error=digitRecognition('train', 'test', 0.001, 5000)
print(h)


