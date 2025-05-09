import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, X, y):
    """
    绘制给定模型的决策边界。

    参数：
    model -- 可调用的模型函数，接受输入数据并返回预测结果。通常是一个训练好的分类器。
    X -- numpy 数组，形状为 (m, 2)，包含 m 个样本的特征，特征数量为 2。
    y -- numpy 数组，形状为 (m, )，包含每个样本的真实标签（0 或 1）。

    函数功能：
    1. 设置 x 和 y 的最小值和最大值，并添加一些填充，以便绘图时有边距。
    2. 生成一个点网格，网格中每个点之间的距离为 h，用于计算决策边界。
    3. 使用模型对整个网格进行预测，得到每个网格点的预测值 Z。
    4. 将预测值 Z 重塑为与网格 xx 的形状相同，以便进行绘图。
    5. 使用等高线图绘制决策边界，并使用不同的颜色表示不同的类别。
    6. 绘制训练样本的散点图，以显示真实数据点在决策边界上的分布。
    7. 显示图形。
    """

    # 设置 x 和 y 的最小值和最大值
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01  # 网格点之间的距离

    # 生成一个点网格
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 对整个网格进行预测
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)  # 重塑预测结果

    # 绘制等高线图和训练样本
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')  # y 轴标签
    plt.xlabel('x1')  # x 轴标签

    # 绘制训练样本的散点图
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()  # 显示图形

