import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-1.0, 1.0, 200)[: , np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise
'''定义两个placeholder'''
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

'''构建神经网络中间层'''
W1 = tf.Variable(tf.random_normal([1, 10]))
b1 = tf.Variable(tf.zeros([1, 10]))
Out1 = tf.matmul(x, W1)+b1
L1 = tf.nn.tanh(Out1)

'''定义输出层'''
W2 = tf.Variable(tf.random_normal([10, 1]))
b2 = tf.Variable(tf.zeros([1, 1]))
Out2 = tf.matmul(L1, W2) + b2
Prediction = tf.nn.tanh(Out2)

loss = tf.reduce_mean(tf.square(y_data-Prediction))
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.Session() as sess:
    # print(sess.run([W1,b1]))
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        sess.run(train, feed_dict={x: x_data, y: y_data})
        print("loss=",  sess.run(loss, feed_dict={x: x_data, y: y_data}))
    pre = sess.run(Prediction, feed_dict={x: x_data})
    plt.figure()
    plt.scatter(x_data,  y_data)

    plt.plot(x_data, pre, 'r-', lw=5)
    plt.show()

