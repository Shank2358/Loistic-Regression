from numpy import *
import matplotlib.pyplot as plt

# 加载数据
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('data/test1.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat

# Sigmoid函数，注意是矩阵运算
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 梯度上升算法
def gradAscent(dataMatIn, classLabels):
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMat)
    alpha = 0.01
    maxCycles = 500
    weights = mat(ones((n, 1)))
    weightsHis = [mat(ones((n, 1)))]  # 权重的记录，主要用于画图
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = labelMat - h
        weights = weights + alpha * dataMat.transpose() * error
        weightsHis.append(weights)
    return weights, weightsHis


# 简单的随机梯度上升，即一次处理一个样本
def stocGradAscent0(dataMatIn, classLabels):
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMat)
    alpha = 0.01
    weights = mat(ones((n, 1)))
    weightsHis = [mat(ones((n, 1)))]  # 权重的记录，主要用于画图
    for i in range(m):
        h = sigmoid(dataMat[i] * weights)
        error = labelMat[i] - h
        weights = weights + alpha * dataMat[i].transpose() * error
        weightsHis.append(weights)
    return weights, weightsHis


# 改进的随机梯度算法
def stocGradAscent1(dataMatIn, classLabels, numIter):
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMat)
    alpha = 0.001
    weights = mat(ones((n, 1)))
    weightsHis = [mat(ones((n, 1)))]  # 权重的记录，主要用于画图
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.001  # 动态调整alpha
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选择样本
            h = sigmoid(dataMat[randIndex] * weights)
            error = labelMat[randIndex] - h
            weights = weights + alpha * dataMat[randIndex].transpose() * error
            del (dataIndex[randIndex])

        weightsHis.append(weights)

    return weights, weightsHis


# 牛顿法
def newton(dataMatIn, classLabels, numIter):
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMat)
    # 对于牛顿法，如果权重初始值设定为1，会出现Hessian矩阵奇异的情况.
    # 原因未知，谁能告诉我
    # 所以这里初始化为0.01
    weights = mat(ones((n, 1))) - 0.99
    weightsHis = [mat(ones((n, 1)) - 0.99)]  # 权重的记录，主要用于画图
    for _ in range(numIter):
        A = eye(m)
        for i in range(m):
            h = sigmoid(dataMat[i] * weights)
            hh = h[0, 0]
            A[i, i] = hh * (1 - hh)

        error = labelMat - sigmoid(dataMat * weights)
        H = dataMat.transpose() * A * dataMat  # Hessian矩阵
        weights = weights + H ** -1 * dataMat.transpose() * error

        weightsHis.append(weights)

    return weights, weightsHis


def plotWeights(w):
    w = array(w)

    def f1(x):
        return w[x, 0, 0]

    def f2(x):
        return w[x, 1, 0]

    def f3(x):
        return w[x, 2, 0]

    k = len(w)
    x = range(0, k, 1)
    plt.plot(x, f1(x), '', x, f2(x), '', x, f3(x), '')
    plt.show()


# 画出分类边界
def plotBestFit(wei):
    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


# 测试
data, labels = loadDataSet()
weights,weightsHis = gradAscent(data, labels)#梯度上升法
weights0, weightsHis0 = stocGradAscent0(data, labels)#简单随机梯度法
weights1, weightsHis1 = stocGradAscent1(data, labels, 500)#改进的随机梯度法
weights3, weightsHis3 = newton(data, labels, 10)#牛顿法
plotBestFit(weights)
print(weights)
plotWeights(weightsHis)
plotBestFit(weights0)
print(weights0)
plotWeights(weightsHis0)
plotBestFit(weights1)
print(weights1)
plotWeights(weightsHis1)
plotBestFit(weights3)
print(weights3)
plotWeights(weightsHis3)

