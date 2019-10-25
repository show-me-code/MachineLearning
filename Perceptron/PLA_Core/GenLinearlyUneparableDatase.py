import numpy as np

# 测试数据为2000个样本点，其中前一千个为均值为[3, 3]的正态分布样本点，标记为-1，后一千个为均值为[7, 7]的正态分布样本点，标记为1，协方差为[[1,0],[0,1]]，数据集有一定概率概率线性可分
mean1=[3,3]
cov1=[[1,0],[0,1]]
mean2=[6,6]
cov2=[[1,0],[0,1]]
X1=np.random.multivariate_normal(mean1,cov1,1000)
X2=np.random.multivariate_normal(mean2,cov2,1000)
y = np.ones(2000)
y[0:1000] = -1
X = np.row_stack((X1,X2))
try:
    f = open("Perceptron\\PLA_Core\\data_set_linearly_unseparable.txt","w")
    # 此处打乱数据集
    for i in np.random.permutation(np.arange(len(y))):
        f.write(str(X[i])[1:-2]+' '+str(y[i]))
        f.write('\n')
    f.close()
except IOError:
    print("Generate dataset error!")