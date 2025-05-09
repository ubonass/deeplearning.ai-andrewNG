import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0, x)

    return s


def he_initialization_normal(n_in, n_out):
    """
    He 初始化（正态分布）
    参数:
        n_in  : 输入神经元数量（前一层维度）
        n_out : 输出神经元数量（当前层维度）
    返回:
        W     : 权重矩阵，形状 (n_in, n_out)
    """
    std = np.sqrt(2.0 / n_in)  # 标准差为 sqrt(2/n_in)
    W = np.random.normal(0, std, size=(n_in, n_out))
    return W


def initialize_parameters(layer_dims):
    """
    初始化全连接神经网络的权重和偏置（He 初始化）
    参数:
        layer_dims : 列表，例如 [input_dim, hidden1_dim, ..., output_dim]
    返回:
        parameters: 包含权重 W1, b1, ..., WL, bL 的字典
    """
    np.random.seed(42)  # 固定随机种子，便于复现
    parameters = {}
    L = len(layer_dims)  # 网络层数（包括输入层和输出层）

    for l in range(1, L):
        n_in = layer_dims[l - 1]
        n_out = layer_dims[l]

        # 初始化权重
        parameters[f'W{l}'] = he_initialization_normal(n_in, n_out)
        # 初始化偏置为零
        parameters[f'b{l}'] = np.zeros((1, n_out))

        # 验证形状
        assert parameters[f'W{l}'].shape == (n_in, n_out), \
            f"权重 W{l} 形状错误，应为 {(n_in, n_out)}"
        assert parameters[f'b{l}'].shape == (1, n_out), \
            f"偏置 b{l} 形状错误，应为 {(1, n_out)}"

    return parameters


# Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
def forward_propagation(X, parameters):
    # X(m, n) , parameters['W1'] (n, )
    W1 = parameters[f'W1']
    b1 = parameters[f'b1']

    Z1 = np.dot(X, parameters['W1']) + parameters['b1']
    A1 = relu(Z1)

    W2 = parameters[f'W2']
    b2 = parameters[f'b2']
    Z2 = np.dot(A1, parameters['W2']) + parameters['b2']
    A2 = relu(Z2)

    W3 = parameters[f'W3']
    b3 = parameters[f'b3']
    Z3 = np.dot(A2, parameters['W3']) + parameters['b3']
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


def compute_cost(Y_hat, Y):
    """
    计算逻辑回归的成本函数。

    参数：
    Y (numpy.ndarray): 标签向量，形状为 (m, 1)

    返回：
    float: 计算得到的成本值，标量，表示模型预测与实际标签之间的差距。
    """
    m = Y.shape[0]
    # 当Y是列向量如(m, 1)时使用Y.T@np.log(A)

    cost = (-1 / m) * (Y.T @ np.log(Y_hat) + (1 - Y).T @ np.log(1 - Y_hat))

    return cost.item()


def compute_cost_with_regularization(Y_hat, Y, parameters, lambd):
    """
    计算带有 L2 正则化的损失函数。

    参数:
    Y_hat -- 前向传播的输出（激活后），形状为 (输出大小, 样本数量)
    Y -- 真实标签向量，形状为 (输出大小, 样本数量)
    parameters -- 包含模型参数的 Python 字典，键为参数名称，值为对应的参数矩阵
    lambd -- 正则化超参数，用于控制正则化项的权重

    返回:
    cost - 正则化损失函数的值 (包含交叉熵和正则化部分)
    """
    m = Y.shape[0]  # 获取样本数量

    # 计算交叉熵部分的成本
    cross_entropy_cost = compute_cost(Y_hat, Y)  # 计算交叉熵成本
    ### START CODE HERE ### (approx. 1 line)
    L2_regularization_cost = 0  # 初始化 L2 正则化成本为 0
    for key in parameters:
        if not key.startswith('W'):  # 只处理以 'W' 开头的参数
            continue
        W = parameters[key]  # 获取当前参数 W
        L2_regularization_cost += np.sum(np.square(W))  # 累加 W 的平方和

    L2_regularization_cost = (lambd / (2 * m)) * L2_regularization_cost  # 计算正则化成本
    ### END CODER HERE ###
    # 总成本是交叉熵成本加上 L2 正则化成本
    cost = cross_entropy_cost + L2_regularization_cost
    return cost  # 返回计算得到的总成本


def backward_propagation(X, Y, cache):
    """
    反向传播算法，计算神经网络的梯度。

    参数:
    X -- 输入数据，形状为 (m, n_x)，其中 m 是样本数量，n_x 是特征数量
    Y -- 真实标签，形状为 (m, n_y)，其中 n_y 是输出层的神经元数量
    cache -- 包含前向传播中计算的值的元组，格式为 (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    返回:
    gradients -- 包含所有梯度的字典，格式为:
                 {
                     "dZ3": dZ3,
                     "dW3": dW3,
                     "db3": db3,
                     "dZ2": dZ2,
                     "dW2": dW2,
                     "db2": db2,
                     "dZ1": dZ1,
                     "dW1": dW1,
                     "db1": db1
                 }
    """

    # 解包缓存中的值
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    m = X.shape[0]  # 样本数量

    # 计算输出层的梯度
    dZ3 = A3 - Y  # 输出层的误差
    dW3 = np.dot(A2.T, dZ3) / m  # 输出层权重的梯度
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m  # 输出层偏置的梯度

    # 计算隐藏层2的梯度
    dZ2 = np.dot(dZ3, W3.T) * (A2 > 0)  # 使用ReLU激活函数的导数
    dW2 = np.dot(A1.T, dZ2) / m  # 隐藏层2权重的梯度

    db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # 隐藏层2偏置的梯度
    # ones_vec = np.ones((dZ2.shape[0]))
    # db2 = np.dot(ones_vec.T, dZ2) / m

    # 计算隐藏层1的梯度
    dZ1 = np.dot(dZ2, W2.T) * (A1 > 0)  # 使用ReLU激活函数的导数
    dW1 = np.dot(X.T, dZ1) / m  # 隐藏层1权重的梯度
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # 隐藏层1偏置的梯度

    # 将所有梯度存储在字典中
    gradients = {
        "dZ3": dZ3,
        "dW3": dW3,
        "db3": db3,
        "dZ2": dZ2,
        "dW2": dW2,
        "db2": db2,
        "dZ1": dZ1,
        "dW1": dW1,
        "db1": db1
    }

    return gradients

def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    实现带有 L2 正则化的反向传播。

    参数:
    X -- 输入数据，形状为 (m, n_x)
    Y -- 真实标签，形状为 (m, n_y)
    cache -- 前向传播的缓存输出
    lambd -- 正则化超参数，标量

    返回:
    gradients -- 包含每个参数、激活和预激活变量的梯度字典
    """
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    gradients = backward_propagation(X, Y, cache)

    m = X.shape[0]

    # 更新权重的梯度以包括正则化项
    gradients["dW3"] += (lambd / m) * W3
    gradients["dW2"] += (lambd / m) * W2
    gradients["dW1"] += (lambd / m) * W1

    return gradients


def update_parameters(parameters, grads, learning_rate):
    """
    使用梯度下降法更新参数

    参数：
    parameters -- 包含模型参数的字典：
                    parameters['W' + str(i)] = Wi  # 权重矩阵
                    parameters['b' + str(i)] = bi   # 偏置向量
    grads -- 包含每个参数梯度的字典：
                    grads['dW' + str(i)] = dWi  # 权重的梯度
                    grads['db' + str(i)] = dbi   # 偏置的梯度
    learning_rate -- 学习率，标量，控制每次更新的步长

    返回：
    parameters -- 包含更新后参数的字典
    """

    # 计算神经网络的层数（假设每层都有一个权重和一个偏置）
    n = len(parameters) // 2  # 参数字典的长度除以 2，得到层数

    # 对每个参数进行更新
    for k in range(n):
        # 更新权重：使用当前权重减去学习率乘以对应的梯度
        parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - learning_rate * grads["dW" + str(k + 1)]

        # 更新偏置：使用当前偏置减去学习率乘以对应的偏置梯度
        parameters["b" + str(k + 1)] = parameters["b" + str(k + 1)] - learning_rate * grads["db" + str(k + 1)]

    # 返回更新后的参数字典
    return parameters


def predict(X, y, parameters):
    """
    该函数用于预测 n 层神经网络的结果。

    参数：
    X -- numpy 数组，形状为 (m, n)，其中 m 是样本数量，n 是特征数量。这个数据集包含你想要标记的示例。
    y -- numpy 数组，形状为 (m, )，包含真实标签（0 或 1）。用于评估模型的准确性。
    parameters -- 字典，包含训练好的模型参数。通常包括权重和偏置项，这些参数用于前向传播计算。

    返回值：
    p -- numpy 数组，形状为 (m, )，包含对给定数据集 X 的预测结果。预测结果为 0 或 1，表示每个样本的分类结果。

    函数功能：
    1. 调用前向传播函数 `forward_propagation`，计算预测值 y_hat 和缓存值 caches。
    2. 将预测值 y_hat 转换为二进制分类结果（0 或 1），使用阈值 0.5。
    3. 计算并打印模型的准确性，即预测结果与真实标签 y 的匹配程度。
    4. 返回预测结果 p。
    """

    # 前向传播
    y_hat, caches = forward_propagation(X, parameters)

    predictions = (y_hat > 0.5).astype(int)
    print(f"Accuracy = {np.mean(predictions == y)}")

    return predictions


def predict_dec(X, parameters):
    """
    该函数用于预测 n 层神经网络的结果。

    参数：
    X -- numpy 数组，形状为 (m, n)，其中 m 是样本数量，n 是特征数量。这个数据集包含你想要标记的示例。
    parameters -- 字典，包含训练好的模型参数。通常包括权重和偏置项，这些参数用于前向传播计算。

    返回值：
    p -- numpy 数组，形状为 (m, )，包含对给定数据集 X 的预测结果。预测结果为 0 或 1，表示每个样本的分类结果。

    函数功能：
    1. 调用前向传播函数 `forward_propagation`，计算预测值 y_hat 和缓存值 caches。
    2. 将预测值 y_hat 转换为二进制分类结果（0 或 1），使用阈值 0.5。
    3. 返回预测结果 p。
    """

    # 前向传播
    y_hat, caches = forward_propagation(X, parameters)
    predictions = (y_hat > 0.5)

    return predictions
