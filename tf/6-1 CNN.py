import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets('data/MNIST', one_hot=True)


batch_size = 128
n_batch = mnist.train.num_examples//batch_size


def weight_init(size):
    initial = tf.truncated_normal(size,stddev=0.1)
    return tf.Variable(initial)

def bias_init(size):
    initial = tf.constant(0.1,shape=size)
    return tf.Variable(initial)

def conv2d(x,w):

    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(X, [-1, 28, 28, 1])



'''初始化第一个卷积层'''
W_conv1 = weight_init([5,5,1,16])#5*5采样窗口,16个卷积核从1个平面抽取特征
b_conv1 = bias_init([16])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

'''第二个卷积层'''
W_conv2 = weight_init([5,5,16,32])#5*5采样窗口,32个卷积核从16个平面抽取特征
b_conv2 = bias_init([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''全连接层'''
W_fc1 = weight_init([7*7*32,1024])
b_fc1 = bias_init([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*32])#平铺维度
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_init([1024,10])
b_fc2 = bias_init([10])

predic = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)


#loss = tf.reduce_mean(tf.square(Y-predic))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=predic))

train = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()

correct = tf.equal(tf.argmax(Y,1),tf.argmax(predic,1))

accury = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(batch_size):
            bx,by = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={X: bx, Y: by, keep_prob: 0.7})

        acc = sess.run(accury, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ",Testing Accuary" + str(acc))

    pre = sess.run(predic, feed_dict={X: mnist.test.images, Y: mnist.test.labels,keep_prob: 1.0})
    print(str(np.argmax(pre[:100],1)))
    print(np.argmax(mnist.test.labels[:100],1))
