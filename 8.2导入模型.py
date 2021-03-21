import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data

# 基于3.2
# 导入数据集
mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

# 输入层 传入数据集接口
x = tf.compat.v1.placeholder(tf.float32, [None, 784])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# 建立输入-输出神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))  # 使用交叉熵
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.2).minimize(loss)  # 最小梯度法训练
# 求准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.compat.v1.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 导入模型
    saver.restore(sess, 'net/my_net.ckpt')
    print("Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

