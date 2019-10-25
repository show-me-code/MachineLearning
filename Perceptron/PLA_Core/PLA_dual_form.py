import numpy as np
import time

# PLA对偶形式实现，X为数据集，y为X对应的标记，y在{-1,,1}中取值
def PLA(X, y, gramMatrix, learning_rate = 0.1):
    # 获取数据的维度，n为样本个数，d为样本维度
    n, d = X.shape
    # 初始化权重，前d位为x的系数，最后一位为截距
    weights = np.array([0.0 for i in range(0, d+1)])
    # 初始化α，其中前n项为w对y[i]*X[i]的线性表示，后一项为截距
    alphas = np.array([0.0 for i in range(0, n+1)])
    steps = 0
    # 算法主体
    while True:
        # 标记这一轮循环是否存在错误分类的样本点
        miss_exist = False
        # 1.遍历样本点
        # 2.判断是否错误分类
        # 3.如果错误分类，将“存在错误”标记置为True，同时利用w←w+η*y[i]*x[i]，b←b+η*y[i]修正，此处将两个式子合为一个表达式，直到该样本点正确分类
        # 4.修正完毕后返回步骤1
        for i in range(0, n):
            while sgn(y[i]*(np.multiply(alphas[0: -1], gramMatrix[i]).dot(y) + alphas[-1])) == -1:
                miss_exist = True
                alphas[i] = alphas[i] + learning_rate
                alphas[-1] = alphas[-1] + learning_rate*y[i]
                # weights = np.append(np.sum(alphas[i]*y[i]*X[i] for i in range(0, n)),alphas[-1])
                # weights_record.append(weights.copy())
                steps += 1
                # print('第'+str(steps)+'次修正')
            if miss_exist:
                continue
        # 遍历完毕且没有错误分类点，完成算法
        if not miss_exist:
            break
    weights = np.append(sum(alphas[i]*y[i]*X[i] for i in range(0, n)),alphas[-1])
    return weights, steps

# 计算Gram矩阵
def calc_Gram(X, n):
    gramMatrix=np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gramMatrix[i][j]=np.dot(X[i],X[j])
    return gramMatrix

# sgn函数
def sgn(num):
    return 1 if num>0 else -1

if __name__ == '__main__':
    array_X = []
    array_y = []
    # 从线性可分数据集data_set_linearly_separable.txt中读取数据
    try:
        f = open("Perceptron\\PLA_Core\\data_set_linearly_separable_1.txt","r")
        while True:
            s = f.readline()
            if not s:
                f.close()
                break
            data = f.readline().rstrip().split()
            array_X.append([float(data[0]),float(data[1])])
            array_y.append(float(data[2]))
    except IOError:
        print('Read file error!')
        exit(0)
    X = np.array(array_X)
    y = np.array(array_y)
    # 获取数据的维度，n为样本个数，d为样本维度
    n, d = X.shape
    # 计算Gram矩阵
    gramMatrix = calc_Gram(X, n)
    # 计算10次的平均运行时间
    starttime = time.time()
    # PLA算法求系数
    for i in range(0, 10):
        learning_rate = 1
        weights, steps = PLA(X, y, gramMatrix, learning_rate)
        print('第'+str(i+1)+'次，w=：'+str(weights[0:-1])+'，b='+str(weights[-1])+'，共迭代'+str(steps)+'次')

    endtime = time.time()
    dtime = (endtime - starttime)/10.0

    print("算法运行10次的平均时间为：%.8s s" % dtime)