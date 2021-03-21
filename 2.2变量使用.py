import tensorflow as tf

# 创建op变量
x = tf.Variable([1, 2])
a = tf.constant([2, 2])
# 减法
sub = tf.subtract(x, a)
# 加法
add = tf.add(x, a)
# 创建变量初始化op
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 运行变量初始化op
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

state = tf.Variable(0, name='counter')
# 创建加法op
new_value = tf.add(state, 1)
# 创建赋值op
update = tf.assign(state, new_value)
# 创建初始化变量op
init = tf.global_variables_initializer()
# 创建会话
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
