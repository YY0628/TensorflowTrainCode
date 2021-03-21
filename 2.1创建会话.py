import tensorflow as tf

# tensorflow里面数据类型都是多维矩阵
# 运算需在session会话中运行，即创建矩阵、矩阵运算后，要添加到默认会话tf.Session()中运行


# 创建常量op ab
a = tf.constant([[2, 4]])
b = tf.constant([[3], [3]])
# ab进行矩阵乘法运算
product = tf.matmul(a, b)
# 使用with as 就不需要sess.close()
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
