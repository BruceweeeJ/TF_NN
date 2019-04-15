import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets('data/MNIST', one_hot=True)


batch_size = 54
n_batch = mnist.train.num_examples//batch_size

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001,tf.float32)

"""前向传播"""
W1 = tf.Variable(xavier_init([784,512]))
b1 = tf.Variable(tf.zeros([512])+0.1)
l1 = tf.nn.relu(tf.matmul(X,W1)+b1)
l1_prob = tf.nn.dropout(l1,keep_prob)

W2 = tf.Variable(xavier_init([512,128]))
b2 = tf.Variable(tf.zeros([128])+0.1)
l2 = tf.nn.relu(tf.matmul(l1_prob,W2)+b2)
l2_prob = tf.nn.dropout(l2,keep_prob)

W3 = tf.Variable(xavier_init([128,10]))
b3 = tf.Variable(tf.zeros([10])+0.1)
#l3 = tf.nn.relu(tf.matmul(l2,W3)+b3)

#W4 = tf.Variable(xavier_init([128,10]))
#b4 = tf.Variable(tf.zeros([10]))
predic = tf.nn.sigmoid(tf.matmul(l2_prob,W3)+b3)


#loss = tf.reduce_mean(tf.square(Y-predic))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=predic))

#train = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

train = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()

correct = tf.equal(tf.argmax(Y,1),tf.argmax(predic,1))

accury = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        sess.run(tf.assign(lr,0.001*(0.95)**epoch))
        for batch in range(n_batch):
            bx,by = mnist.train.next_batch(batch_size)

            sess.run(train,feed_dict={X:bx,Y:by,keep_prob:1})

        test_acc = sess.run(accury,feed_dict={X:mnist.test.images,Y:mnist.test.labels,keep_prob:1})
        train_acc = sess.run(accury, feed_dict={X: mnist.train.images, Y: mnist.train.labels,keep_prob:1})
        print("Iter "+ str(epoch)+",Testing Accuary: " +str(test_acc)+",Training Accuary: " +str(train_acc))
    pre = sess.run(predic, feed_dict={X: mnist.test.images, Y: mnist.test.labels,keep_prob:1})
    print(str(np.argmax(pre[:100],1)))
    print(np.argmax(mnist.test.labels[:100],1))



