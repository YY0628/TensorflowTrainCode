import tensorflow as tf

# Fetch
# 一次运行多个会话
input1 = tf.constant(3.0)
input2 = tf.constant(4.0)
input3 = tf.constant(5.0)

add = tf.add(input1,input2)
mul = tf.multiply(input1,add)

with tf.Session() as sess:
    result = sess.run([add,mul])
    print(result)

# Feed 占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[8.0],input2:[7.0]}))

