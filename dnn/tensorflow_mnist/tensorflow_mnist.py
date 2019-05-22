# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.contrib.layers import fully_connected



# load data and configure parameters
mnist = input_data.read_data_sets("MNIST_data/")
lr = 0.1
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs= 10
epochs = 40
batch_size = 50
iterations = mnist.train.num_examples // batch_size


# define graph
x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.name_scope('dnn'):
    hidden1 = fully_connected(x, n_hidden1, activation_fn=tf.nn.relu, scope='hidden1')
    hidden2 = fully_connected(hidden1, n_hidden2, activation_fn=tf.nn.relu, scope='hidden2')
    logits = fully_connected(hidden2, n_outputs, activation_fn=None, scope='logits')
    
with tf.name_scope('loss'):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(entropy, name='loss')
    
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train = optimizer.minimize(loss)    # apply_gradients(compute_gradients)
    
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    acc = tf.reduce_mean(tf.cast(correct, tf.float32), name='acc')

init = tf.global_variables_initializer()
saver = tf.train.Saver()    # defalt: save all variables 



# call graph
with tf.Session() as sess:
    init.run()  # sess.run(init)
    for epoch in range(epochs):
        acc_train = 0.0
        for iteration in range(iterations):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _, acc_batch = sess.run([train, acc], feed_dict={x:x_batch, y:y_batch})
            acc_train += acc_batch
        print(epoch, 'Train_acc:', acc_train/iterations) 
        
    saver.save(sess, 'model/dnn.ckpt')
            
with tf.Session() as sess:
    saver.restore(sess, 'model/dnn.ckpt')
    acc_test = acc.eval(feed_dict={x:mnist.test.images, y: mnist.test.labels})
    print("Acc_test:{0}".format(acc_test))
            
            