# Language model
# output all vocabulary
# for testing

import tensorflow as tf
import pickle
import numpy as np
import time

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config = config))

class Model(object):
	def __init__(self):
		self.data_size 	= 636000 	# 435000 or 437600
		self.batch_size = 200
		self.num_steps 	= 30 		# #(neurons) of input layer 1
		self.dim_size 	= 11792 	# #(neurons) of input layer 2
		self.num_layers = 2 
		self.hidden_size1 = 800 	# unknown, need to be ckeck, 200, 650, 1500
		self.hidden_size2 = 1800
		self.hidden_size3 = 2400 	# -- no
		self.vocab_size = 3591 		# #(neurons) of output layer (class)
		self.is_train 	= 630000 	# 430000 or 420000
		self.v1 = 630000 			# 430000
		self.v2 = 636000 			# 435000
		self.t1 = 0
		self.batch_num = self.is_train / self.batch_size # how many batch, = 2100
		self.valid_num = (self.data_size - self.is_train) / self.batch_size
		self.valid_size = self.data_size - self.is_train
						# how many validation batch
		self.from_last = 0
		self.beta1 = 0.00
		self.beta2 = 0.00
		self.drop = 0.4
		self.lr = 100e-5 
		self.epoch = 3151+1500
		self.small = 0
		self.min_loss_v = 1.0
		self.hotx_zero = np.zeros((self.batch_size, 30, self.dim_size))
		self.hoty_zero = np.zeros((self.batch_size, self.vocab_size))
		self.path = "../"
		self.n = 1040
		print "--- lr", self.lr, "b", self.beta1, self.beta2

	def lstm_layer_multi(self, X): # dropout set at in/ out of rnn!!!
		X = tf.reshape(X, [-1, self.dim_size])
		X_in = (tf.matmul(X, self.w1) + self.b1)
		X_in = tf.reshape(X_in, [-1, self.num_steps, self.hidden_size1])
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size1, forget_bias = 1.0, state_is_tuple = True)
		lstm_stack = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple = True)
		init_state = lstm_stack.zero_state(self.bs, dtype = tf.float32)
		outputs, final_state = tf.nn.dynamic_rnn(lstm_stack, X_in, initial_state = init_state, time_major = False)
		outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2])) # states is the last outputs
		results = tf.matmul(outputs[-1], self.w4) + self.b4
		return results
	
	def set_nn_parameter(self):
  		print "Setting parameter"
  		self.w1 = tf.Variable(tf.random_normal([self.dim_size, self.hidden_size1]))
		self.b1 = tf.Variable(tf.constant(0.1, shape = [self.hidden_size1, ]))

		self.w2 = tf.Variable(tf.random_normal([self.hidden_size1, self.hidden_size2]))
		self.b2 = tf.Variable(tf.constant(0.1, shape = [self.hidden_size2, ]))

		self.w3 = tf.Variable(tf.random_normal([self.hidden_size2, self.hidden_size3]))
		self.b3 = tf.Variable(tf.constant(0.1, shape = [self.hidden_size3, ]))

		self.w4 = tf.Variable(tf.random_normal([self.hidden_size1, self.vocab_size]))
		self.b4 = tf.Variable(tf.constant(0.1, shape = [self.vocab_size, ]))

		self.x_nn = tf.placeholder(tf.float32, [None, self.num_steps, self.dim_size])
		self.y_nn = tf.placeholder(tf.float32, [None, self.vocab_size])
		self.keep_prob = tf.placeholder("float")
		self.bs = tf.placeholder(dtype = tf.int32)

	def hotx(self, arr):
		new_arr = np.zeros((1, 30, self.dim_size), dtype = int)
		new_arr[0][np.arange(arr.shape[1]), arr[0]] = 1
		return new_arr

	def hoty(self, arr):
		new_arr = np.zeros((self.batch_size, self.vocab_size))
		new_arr[np.arange(arr.shape[0]), arr] = 1
		return new_arr

	def get_ans_id(self, n):
		if   n == 0: return "a"
		elif n == 1: return "b"
		elif n == 2: return "c"
		elif n == 3: return "d"
		elif n == 4: return "e"
		else: return "error!"

	def load_test(self):
		print "Loading data"
		self.temp = pickle.load(open(self.path + "model/x_test_hot", "rb"))
		self.y_test = pickle.load(open(self.path + "model/y_test_hot", "rb"))
		self.x_test = np.empty([1040, 30], dtype = int)
		for i in range(self.n):
			for j in range(30):
				self.x_test[i][j] = int(self.temp[i][j])
		
	def compute_answer(self, sess, pred):
		self.load_test()
		back = 2 # consider the prob of latter 2 words
		
		prob = np.empty([self.n, 5], dtype = float)
		x_t1 = np.empty([1, 30], dtype = int)
		y_t1_id = np.empty(5, dtype = int)
		for i in range(self.n):
			x_t1[0] = self.x_test[i] # for only one testing data
			for j in range(5): y_t1_id[j] = int(self.y_test[i][j])
			ans = sess.run(pred, feed_dict = {self.x_nn: self.hotx(x_t1), self.bs: 1, self.keep_prob: 1}) 
			for j in range(5): prob[i][j] = ans[0][y_t1_id[j]] # input 5 probable y id

		for i in range(5):
			for j in range(5):
				print "%.3f" % prob[i][j],
			print ""

		self.output_answer(prob)

	def test_nn(self):
  		pred = self.lstm_layer_multi(self.x_nn)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y_nn, logits = pred))
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
		optimizer = tf.train.AdamOptimizer(self.lr)
		train_step = optimizer.apply_gradients(zip(grads, tvars))
		correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y_nn, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		with tf.Session() as sess:
			saver.restore(sess, self.path + "/model/model.ckpt")
		  	print("Model restored.")
			sess.run(init)
			self.compute_answer(sess, pred)

def main():
	model = Model()
	model.load_test()
	model.set_nn_parameter()
	model.test_nn()
	print "\n Compile successfully!"

if __name__ == '__main__':
	main()


