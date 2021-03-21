import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 制作样本数据，创建随机数，并升维至[200,1]
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 创建训练数据，格式与样本数据一样
x = tf.compat.v1.placeholder(tf.float32, [None, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 1])

# 神经网络，输入层1个神经元，中间层10个神经元，输出层1个神经元
# 中间层权值与偏置结构为[1,10]
Weights_L1 = tf.Variable(tf.random.normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# 激活函数为正切函数
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 输出层权值结构为[10,1] 偏置结构为[1,1]因为偏置做加法需要矩阵结构一致，而权值做矩阵乘法
Weights_L2 = tf.Variable(tf.random.normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 二次损失函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 以最小梯度训练
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 训练2000次神经网络
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    # 上述训练完成后Weights_L1,biases_L1,Weights_L2,biases_L2以训练完毕
    # 使用神经网络
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
