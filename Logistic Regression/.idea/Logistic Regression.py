# -*- coding: utf-8 -*-
from numpy import *
from os import listdir

# data=[]
# label=[]
def loadData(direction):
    print(direction)
    dataArray = []
    labelArray = []
    trainfileList = listdir(direction)
    m = len(trainfileList)
    for i in range(m):
      filename = trainfileList[i]
      fr = open('%s/%s' % (direction, filename))
      for line in fr.readlines():
          lineArr = line.strip().split()
          dataArray.append([float(lineArr[0]),float(lineArr[1])])
          labelArray.append([int(lineArr[2])])
      fr.close()
    # data=transpose(dataArray)
    # label = transpose(labelArray)
    # print(data)
    # print(label)
    return dataArray, labelArray

# sigmoid(inX)函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 用梯度下降法计算得到回归系数，alpha是步长，maxCycles是迭代步数。
# def gradAscent(dataArray, labelArray, alpha, maxCycles):
#     dataMat = mat(dataArray)  # size:m*n
#     labelMat = mat(labelArray)  # size:m*1
#     m, n = shape(dataMat)
#     weigh = ones((n, 1))
#     for i in range(maxCycles):
#         h = sigmoid(dataMat * weigh)
#         error = labelMat - h  # size:m*1
#         weigh = weigh + alpha * dataMat.transpose() * error
#     return weigh
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
    h = sigmoid(dataMat * weigh)  # size:m*1
    print(h)
    m = len(h)
    error = 0.0
    for i in range(m):
        if int(h[i]) > 0.5:
            print(int(labelMat[i]), 'is classfied as: 1')
            if int(labelMat[i]) != 1:
                error += 1
                print('error')
        else:
            print(int(labelMat[i]), 'is classfied as: 0')
            if int(labelMat[i]) != 0:
                error += 1
                print('error')
    print('error rate is:', '%.4f' % (error / m))


"""
用loadData函数从train里面读取训练数据，接着根据这些数据，用gradAscent函数得出参数weigh，最后就可以用拟
合参数weigh来分类了。
"""

def digitRecognition(trainDir, testDir, alpha, maxCycles):
    data, label = loadData(trainDir)
    weigh = gradAscent(data, label, alpha, maxCycles)
    classfy(testDir, weigh)


# 运行函数
digitRecognition('train', 'test', 0.01, 50)

