import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# (tensorflow115-gpu) H:\>tensorboard --logdir==H:\test\source\tfLearn\logs
# 谷歌浏览器输入 http://localhost:6006/
# 导入数据集
mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)
# 设置训练数据集一批 batch_size 个
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

with tf.name_scope('input'):
    # 输入层 传入数据集接口
    x = tf.compat.v1.placeholder(tf.float32, [None, 784])
    y = tf.compat.v1.placeholder(tf.float32, [None, 10])

with tf.name_scope('layer'):
    # 建立输入-输出神经网络
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784, 10]))
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]))
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(tf.matmul(x, W) + b)

with tf.name_scope('loss'):
    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
with tf.name_scope('train'):
    # 分类器
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.name_scope('accuracy'):
    # 求准确率
    with tf.name_scope('corect_predtiction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))    # argmax()返回最大值所在序列，prediction返回最大概率序列，
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))          # y则返回1所在序列 ；当前后序列相等，则返回1，最终返回01列表
                                                                                # 对列表值求平均，即正确率
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(1):
        for batch in range(n_batch):
            # 读取训练数据并喂入
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        # 使用测试样本计算正确率
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter" + str(epoch) + ",Test Accuracy" + str(acc))
