import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def perceptron(X, y, learning_rate = 0.1):
    # 获取数据的维度，n为样本个数，d为样本维度
    n, d = X.shape
    weights = np.array([np.random.rand() for i in range(0, d+1)])
    weights_record = []
    weights_record.append(weights.copy())
    steps = 0
    while True:
        miss_exist = False
        for i in range(0, n):
            while sgn(y[i]*(weights[0:-1].dot(X[i,:])+weights[-1])) == -1:
                miss_exist = True
                weights = weights + learning_rate*y[i]*np.append(X[i,:],1)
                weights_record.append(weights.copy())
                # weights[-1] = weights[-1] + learning_rate*y[i]
                steps += 1
                print('第'+str(steps)+'次修正，w=：'+str(weights[0:-1])+'，b='+str(weights[-1]))
            if miss_exist:
                continue
        if not miss_exist:
            break
    return weights, weights_record

def sgn(num):
    return 1 if num>0 else -1

# X = np.array([[1, 3], [2, -1], [2, 0], [0.2, 1], [3, 1.5]])
# y = np.array([1, -1, 1, 1, -1])

 
#二维测试数据生成
mean1=[0,0]
cov1=[[1,0],[0,1]]
mean2=[3,3]
cov2=[[1,0],[0,1]]
X1=np.random.multivariate_normal(mean1,cov1,50)
X2=np.random.multivariate_normal(mean2,cov2,50)
#绘制散点图
fig, ax = plt.subplots()
plt.scatter(X1[:,0],X1[:,1])
plt.scatter(X2[:,0],X2[:,1])
X = np.row_stack((X1,X2))
y = np.ones(100)
y[0:50] = -1

#PLA算法求系数
weights, weights_record = perceptron(X, y, 0.1)
x = np.arange(-5,5,0.01)
w0 = weights_record[0]

line, = ax.plot([], [], color='red', animated=False)

def init():
    ax.set_xlim(-2,6)
    ax.set_ylim(-2,6)
    return line,
 
def animate(i):
    if i >= len(weights_record):
        title = ax.text(0.5,0.95, 'Fixed '+str(len(weights_record)-1)+' times', bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center") 
        return [line]+[title]
    weights = weights_record[i]
    title = ax.text(0.5,0.95, 'Fixed '+str(i)+' times', bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center") 
    if(weights[1]!=0.0):
        line.set_ydata((-weights[0]*x-weights[2])/weights[1])
    else:
        line.set_data(-weights[2]/weights[0],np.arange(-5,5,0.01))  # update the data.
    return [line]+[title]
ani = animation.FuncAnimation(fig, animate, init_func=init, interval=100, blit=True)
line, = ax.plot(x, (-weights[0]*x-weights[2])/weights[1])
plt.show()