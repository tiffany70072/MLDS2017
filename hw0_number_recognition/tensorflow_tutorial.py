'''
Tutorial of TensorFlow by TA in MLDS lecture
'''

import tensorflow as tf
import numpy as np
import random
import sys

# data preparing
N = 10
D = 2
x = np.zeros((N, D), dtype = float)
y = np.zeros((N, 1), dtype = float)

for i in range(N):
	for j in range(D):
		x[i][j] = i*1 + random.random() - 0.5
	y[i] = 3*i + random.random() - 0.5

print "x = ", x[:10]
print "y = ", y[:10]

# session
lr = 1E-4

nn_x = tf.placeholder(tf.float32, [None, D])
nn_y = tf.placeholder(tf.float32, [None, 1])
w1   = tf.Variable(tf.random_uniform([D, 30], -1, 1))
b1   = tf.Variable(tf.zeros([1, 30]))
w2   = tf.Variable(tf.random_uniform([30, 1], -1, 1))
b2   = tf.Variable(tf.zeros([1, 1]))
hidd = tf.matmul(nn_x, w1) + b1
Y    = tf.matmul(hidd, w2) + b2
#Y    = tf.nn.dropout(Y, 0.7) # 0.3 or 0.7
#Y    = tf.matmul(Y, self.W2)+ self.b2
loss = tf.reduce_mean(tf.square(nn_y - Y))
# train   = tf.train.AdamOptimizer(lr).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
initial = tf.global_variables_initializer()

# training
sess = tf.Session()
sess.run(initial)
for i in range(300):
	_, loss_val = sess.run([train_step, loss], feed_dict={nn_x: x, nn_y: y})
	if i/30 == 0: print 'loss = ', loss_val

# predict
x_test = np.array([(1, 1), (3, 3)], dtype = float)

print "x_test = ", x_test
x_test_para = x_test
Y_hat = sess.run(Y, feed_dict={nn_x: x_test_para}) 
print Y_hat
