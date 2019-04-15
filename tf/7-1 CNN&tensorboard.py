import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets('data/MNIST', one_hot=True)


batch_size = 100
n_batch = mnist.train.num_examples//batch_size

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)

def weight_init(size,name):
    initial = tf.truncated_normal(size,stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_init(size,name):
    initial = tf.constant(0.1,shape=size)
    return tf.Variable(initial,name=name)

def conv2d(x,w):

    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    with tf.name_scope('X_image'):
        x_image = tf.reshape(X, [-1, 28, 28, 1],name='X_image')

'''初始化第一个卷积层'''
with tf.name_scope('Conv1'):
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_init([5,5,1,32],name='W_conv1')#5*5采样窗口,32个卷积核从1个平面抽取特征
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_init([32],name='b_conv1')
    with tf.name_scope('h_conv1'):
        h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

'''第二个卷积层'''
with tf.name_scope('Conv2'):
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_init([5,5,32,64],name='W_conv2')#5*5采样窗口,64个卷积核从32个平面抽取特征
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_init([64],name='b_conv2')
    with tf.name_scope('h_conv2'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

'''全连接层'''
with tf.name_scope('fc1'):
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_init([7*7*64,1024],name='W_fc1')
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_init([1024],name='b_fc1')
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])#平铺维度
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
with tf.name_scope('fc2'):
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_init([1024,10],name='W_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_init([10],name='b_fc2')
    with tf.name_scope('softmax'):
        predic = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)


#cross_entropy = tf.reduce_mean(tf.square(Y-predic))
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=predic),name='cross_entropy')
    tf.summary.scalar('cross_entropy',cross_entropy)
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(predic,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train',sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)
    for i in range(1001):
        bx,by = mnist.train.next_batch(batch_size)
        sess.run(train,feed_dict={X:bx,Y:by,keep_prob:0.5})
        summary = sess.run(merged,feed_dict={X:bx,Y:by,keep_prob:1.0})
        train_writer.add_summary(summary,i)

        bx, by = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={X: bx, Y: by, keep_prob: 1.0})
        test_writer.add_summary(summary, i)
        if i%100 == 0:
            test_acc = sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels,keep_prob:1.0})
            train_acc = sess.run(accuracy,feed_dict={X:mnist.train.images[:10000],Y:mnist.train.labels[:10000],keep_prob:1.0})
            print("Iter " + str(i) + ", Testing Accuracy=" + str(test_acc)+ ", Training Accuracy=" + str(train_acc))