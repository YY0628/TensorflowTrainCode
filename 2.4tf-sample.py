import tensorflow as tf
import numpy as np

# 使用numoy生成100个随机点
x_data = np.random.rand(100)
y_data = 0.2 * x_data + 0.2

# 构造一个线性模型
k = tf.Variable(0.)
b = tf.Variable(0.)
y = k * x_data + b

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - y_data))
# 定义一个梯度下降发来进行训练的优化器 0.2为学习速率&步长
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.2)
# 以最小代价函数的目标进行训练
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run([k, b]))
