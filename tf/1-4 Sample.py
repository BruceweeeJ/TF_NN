import tensorflow as tf
import numpy as np

x_data = np.random.rand(100)
y_data = x_data*5 + 3.2

'''创建模型'''
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b
'''二次代价函数'''
loss = tf.reduce_mean(tf.square(y_data-y))
'''梯度下降'''
optimizer = tf.train.AdamOptimizer(0.2)
'''最小化代价函数'''
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session()as sess:
    sess.run(init)
    for step in range(20001):
        sess.run(train)
        if step%200==0 :
            print("loss=",sess.run([k,b]))