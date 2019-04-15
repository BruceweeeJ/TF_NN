import tensorflow as tf
m1 = tf.Variable([1,2])
a = tf.constant([3,3])
sub = tf.subtract(m1,a)
add = tf.add(m1,sub)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

state = tf.Variable(0)
#创建一个op
new = tf.add(state,1)
update = tf.assign(state,new)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
