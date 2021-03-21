import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data

# 导入数据集
mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)
# 设置训练数据集一批 batch_size 个
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

# 传入数据集接口
x = tf.compat.v1.placeholder(tf.float32, [None, 784])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])
keep_prob = tf.compat.v1.placeholder(tf.float32)

# 使用dropout以防止过拟合，keep_prob表示使用神经元的比例
W1 = tf.Variable(tf.truncated_normal([784, 1000], stddev=0.1))      # 截断的随机正态分布
b1 = tf.Variable(tf.zeros([1000]) + 0.1)        # 偏置+0.1
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([1000, 1000], stddev=0.1))
b2 = tf.Variable(tf.zeros([1000]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

# 选取损失函数
# 使用交叉熵，使得距离目标远的更快速，距离目标近的慢速
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
# loss = tf.reduce_mean(tf.square(prediction - y))    # 二次损失函数，
# 优化器 标准梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 求准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))    # argmax()返回最大值所在序列，prediction返回最大概率序列，
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))          # y则返回1所在序列 ；当前后序列相等，则返回1，最终返回01列表
                                                                            # 对列表值求平均，即正确率
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    for epoch in range(31):
        for batch in range(n_batch):
            # 读取训练数据并喂入
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 训练时，使70%神经元工作
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.6})
        # 使用测试样本计算正确率
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})

        print("Iter" + str(epoch) + ",Test Accuracy：" + str(test_acc) + ",Training Accuracy：" + str(train_acc))
