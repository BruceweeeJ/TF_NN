import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets('data/MNIST', one_hot=True)

#命名空间
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, 784],name='x-input')
    Y = tf.placeholder(tf.float32, [None, 10],name='y-input')

batch_size = 54
n_batch = mnist.train.num_examples//batch_size

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)



"""前向传播"""
W1 = tf.Variable(xavier_init([784,256]))
b1 = tf.Variable(tf.zeros([256]))
l1 = tf.nn.relu(tf.matmul(X,W1)+b1)
W2 = tf.Variable(xavier_init([256,10]))
b2 = tf.Variable(tf.zeros([10]))

predic = tf.nn.sigmoid(tf.matmul(l1,W2)+b2)


loss = tf.reduce_mean(tf.square(Y-predic))

train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

correct = tf.equal(tf.argmax(Y,1),tf.argmax(predic,1))

accury = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    writer.close()
    for epoch in range(1):
        for batch in range(n_batch):
            bx,by = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={X:bx,Y:by})

        acc = sess.run(accury,feed_dict={X:mnist.test.images,Y:mnist.test.labels})
        print("Iter "+ str(epoch)+",Testing Accuary" +str(acc))
    pre = sess.run(predic, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print(str(np.argmax(pre[:100],1)))
    print(np.argmax(mnist.test.labels[:100],1))



