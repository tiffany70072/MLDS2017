import numpy as np
import tensorflow as tf

def mul_hot(arr, data_size, max_idx1, max_idx2):
	new_arr = np.zeros((data_size, max_idx1+max_idx2), dtype = int)
	for i in range(data_size):
		new_arr[i][arr[i][1]] = 1
		new_arr[i][arr[i][2] + max_idx1] = 1
	return new_arr

def conv2d_transpose(x, w_dim, output_dim, name):
	with tf.variable_scope(name):
		w = tf.get_variable('w', w_dim, initializer = tf.random_normal_initializer(stddev = 0.02))
		deconv = tf.nn.conv2d_transpose(x, w, output_shape = output_dim, strides = [1, 2, 2, 1])
		b = tf.get_variable('b', [output_dim[-1]], initializer = tf.constant_initializer(0.0))
		deconv = tf.nn.bias_add(deconv, b)
		return deconv
		
def conv2d(img, dim, name):
	with tf.variable_scope(name):
		w = tf.get_variable('w', dim, initializer = tf.random_normal_initializer(stddev = 0.02))
		conv = tf.nn.conv2d(img, w, strides=[1, 2, 2, 1], padding='SAME')
		b = tf.get_variable('b', [dim[-1]], initializer = tf.constant_initializer(0.0)) # 0.1
		conv = tf.nn.bias_add(conv, b)
		return conv

def matmul(x, input_dim, output_dim, name):
	with tf.variable_scope(name):
		#w = tf.get_variable(tf.random_normal([input_dim, output_dim]))
		#b = tf.get_variable(tf.constant(0.1, shape = [output_dim, ]))
		w = tf.get_variable('w', [input_dim, output_dim], initializer = tf.random_normal_initializer(stddev = 0.02))
		b = tf.get_variable('b', [output_dim, ], initializer = tf.constant_initializer(0.1))
		return tf.matmul(x, w) + b

def lrelu(x):
	leak = 0.2
	with tf.variable_scope("lrelu"):
		#f1 = 0.5 * (1 + leak)
		#f2 = 0.5 * (1 - leak)
		#return f1 * x + f2 * abs(x)
		return tf.maximum(leak*x, x)

def batch_norm(x, name):
	y = tf.contrib.layers.batch_norm(x, decay = 0.9, updates_collections = None, epsilon = 1e-5, scale = True, scope = name)
	return y
