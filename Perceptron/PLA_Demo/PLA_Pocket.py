import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PLA_Pocket算法实现，X为数据集，y为X对应的标记，y在{-1,,1}中取值
def PLA(X, y, learning_rate = 0.1, max_step = 1000):
    # 获取数据的维度，n为样本个数，d为样本维度
    n, d = X.shape
    # 初始化权重，前d位为x的系数，最后一位为截距
    weights = np.array([np.random.rand() for i in range(0, d+1)])
    # 用于记录权重向量
    weights_record = []
    weights_record.append(weights.copy())
    pocket_weights = weights.copy()
    pocket_accurate_rate = 0
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
            while sgn(y[i]*(weights[0:-1].dot(X[i,:])+weights[-1])) == -1:
                miss_exist = True
                weights = weights + learning_rate*y[i]*np.append(X[i,:],1)
                weights_record.append(weights.copy())
                steps += 1
                current_accurate_rate = get_accurate_rate(X, y, weights)
                print('第'+str(steps)+'次修正，w=：'+str(weights[0:-1])+'，b='+str(weights[-1])+'，正确率：'+str(current_accurate_rate)+'，最高正确率：'+str(pocket_accurate_rate))
                if current_accurate_rate >= pocket_accurate_rate:
                    pocket_accurate_rate = current_accurate_rate
                    pocket_weights = weights
                if steps == max_step:
                    return pocket_weights, weights_record, pocket_accurate_rate
            if miss_exist:
                continue
        # 遍历完毕且没有错误分类点，完成算法
        if not miss_exist:
            break
    return weights, weights_record, pocket_accurate_rate

# sgn函数
def sgn(num):
    return 1 if num>0 else -1

# 计算当前分类的精确度
def get_accurate_rate(X, y, weights):
    mistakes = 0.0
    n,m = X.shape
    for i in range(0, n):
        if sgn(y[i]*(weights[0:-1].dot(X[i,:])+weights[-1])) == -1:
            mistakes += 1
    return (n-mistakes)/n




# 动画初始化
def init():
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    return line,

# 动画帧，展示更新次数和线条 
def animate(i):
    if i >= max_steps:
        line.set_ydata((-weights[0]*x_range-weights[2])/weights[1])
        title = ax.text(0.5,0.85, 'Fixed '+str(max_steps)+' times\nThis dataset seems not linearly separable\nShowing the most accurate result, accurate_rate='+str(pocket_accurate_rate), bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center") 
        return [line]+[title]
    elif i >= len(weights_record):
        line.set_ydata((-weights[0]*x_range-weights[2])/weights[1])
        title = ax.text(0.5,0.95, 'Fixed '+str(len(weights_record)-1)+' times', bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center") 
        return [line]+[title]
    w = weights_record[i]
    title = ax.text(0.5,0.95, 'Fixed '+str(i)+' times', bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center") 
    if(w[1]!=0.0):
        line.set_ydata((-w[0]*x_range-w[2])/w[1])
    else:
        line.set_data(-w[2]/w[0],np.arange(-5,5,0.01))  # update the data.
    return [line]+[title]

if __name__ == '__main__':
    # 调试数据
    # X = np.array([[1, 3], [2, -1], [2, 0], [0.2, 1], [3, 1.5]])
    # y = np.array([1, -1, 1, 1, -1])


    # 测试数据为100个样本点，其中前五十个为均值为[3.5, 3.5]的正态分布样本点，标记为-1，后五十个为均值为[6, 6]的正态分布样本点，标记为1，协方差为[[1,0],[0,1]]，数据集极大概率线性不可分
    mean1=[3.5,3.5]
    cov1=[[1,0],[0,1]]
    mean2=[6,6]
    cov2=[[1,0],[0,1]]
    X1=np.random.multivariate_normal(mean1,cov1,50)
    X2=np.random.multivariate_normal(mean2,cov2,50)
    X = np.row_stack((X1,X2))
    y = np.ones(100)
    y[0:50] = -1
    # 绘制样本集
    fig, ax = plt.subplots()
    plt.scatter(X1[:,0],X1[:,1])
    plt.scatter(X2[:,0],X2[:,1])

    # PLA算法求系数
    max_steps = 1000
    weights, weights_record, pocket_accurate_rate = PLA(X, y, 1, max_steps)
    print('最终结果，w=：'+str(weights[0:-1])+'，b='+str(weights[-1])+'，正确率：'+str(pocket_accurate_rate))
    x_range = np.arange(0,10,0.01)

    line, = ax.plot([], [], color='red', animated=False)
    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=10, blit=True)
    line, = ax.plot(x_range, (-weights[0]*x_range-weights[2])/weights[1])
    plt.show()
