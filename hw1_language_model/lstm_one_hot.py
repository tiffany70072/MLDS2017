# Language model
# output all vocabulary
 
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
		self.hidden_size1 = 800 	# unknown, 200, 650, 1500
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

	def load_x_train(self):
		print "Loading x_train"
		self.x = pickle.load(open(self.path + "model/x_train_260", "rb")) # 300-dim word vector 
		self.x = self.x[:self.data_size]
	
	def load_y_train(self):
		print "Loading y_train"
		self.y_train = pickle.load(open(self.path + "model/y_train_260", "rb"))
		self.y_train = self.y_train[:self.data_size]

	def cut_valid(self):
		self.x, self.y_train = self.shuffle_in_unison(self.x, self.y_train)

		self.x_valid = self.x[self.v1:self.v2]
		self.x = self.x[self.t1:self.t1+self.is_train]
		print "x = ", self.x.shape
		self.y_valid = self.y_train[self.v1:self.v2]
		self.y_train = self.y_train[self.t1:self.t1+self.is_train]
		print "y = ", self.y_train.shape

		self.x, self.y_train = self.shuffle_in_unison(self.x, self.y_train)
		
	def shuffle_in_unison(self, a, b):
	    print "Shuffling"
	    assert len(a) == len(b)
	    np.random.seed(0)
	    shuffled_a = np.empty(a.shape, dtype = a.dtype)
	    shuffled_b = np.empty(b.shape, dtype = b.dtype)
	    permutation = np.random.permutation(len(a))
	    for old_index, new_index in enumerate(permutation):
	        shuffled_a[new_index] = a[old_index]
	        shuffled_b[new_index] = b[old_index]
	    return shuffled_a, shuffled_b

	def lstm_layer(self, X, w1, b1, w2, b2):
		X = tf.reshape(X, [-1, self.dim_size])
		X_in = tf.matmul(X, w1) + b1
		X_in = tf.reshape(X_in, [-1, self.num_steps, self.hidden_size])
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias = 1.0, state_is_tuple = True)
		#lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = self.drop)
		init_state = lstm_cell.zero_state(self.bs, dtype = tf.float32)
		outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state = init_state, time_major = False)
		outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2])) # states is the last outputs
		results = tf.nn.relu(tf.matmul(outputs[-1], w2) + b2)
		results = tf.nn.dropout(results, self.keep_prob)
		return results

	def lstm_layer_multi(self, X): # dropout set at in/ out of rnn!!!
		X = tf.reshape(X, [-1, self.dim_size])
		X_in = (tf.matmul(X, self.w1) + self.b1)
		#X_in = tf.nn.relu(tf.matmul(X_in, self.w2) + self.b2)
		X_in = tf.reshape(X_in, [-1, self.num_steps, self.hidden_size1])
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size1, forget_bias = 1.0, state_is_tuple = True)
		lstm_stack = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple = True)
		init_state = lstm_stack.zero_state(self.bs, dtype = tf.float32)
		outputs, final_state = tf.nn.dynamic_rnn(lstm_stack, X_in, initial_state = init_state, time_major = False)
		outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2])) # states is the last outputs
		#results = (tf.matmul(outputs[-1], self.w2) + self.b2)
		#results = tf.matmul(results, self.w4) + self.b4
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

	def cut_batch(self, step):
		self.batch_id = step % self.batch_num
		begin = self.batch_size * self.batch_id
		end = self.batch_size * (self.batch_id + 1)
		return begin, end

	def hotx(self, arr):
		new_arr = np.zeros((self.batch_size, 30, self.dim_size))
		for i in range(self.batch_size):
			new_arr[i][np.arange(arr.shape[1]), arr[i]] = 1
		return new_arr

	def hoty(self, arr):
		new_arr = np.zeros((self.batch_size, self.vocab_size))
		new_arr[np.arange(arr.shape[0]), arr] = 1
		return new_arr

	def output_accuracy(self, loss, accuracy):
		acc2 = 0
		loss2 = 0
		n = 100
		for i in range(self.batch_num/n):
			begin = self.batch_size * n*i
			end = self.batch_size * (n*i + 1)
			temp = self.sess.run(accuracy, feed_dict = {self.x_nn: self.hotx(self.x[begin:end]), 
				self.y_nn: self.hoty(self.y_train[begin:end]), self.bs: self.batch_size, self.keep_prob: self.drop}),
			acc2 += temp[0]
			loss2 += self.sess.run(loss, feed_dict = {self.x_nn: self.hotx(self.x[begin:end]), 
				self.y_nn: self.hoty(self.y_train[begin:end]),
				self.bs: self.batch_size, self.keep_prob: self.drop})
			
		acc2 /= (self.batch_num/n)
		loss2 /= (self.batch_num/n)
	
		print ", acc: %.3f" % acc2,
		print ", loss: %.4f" % (loss2 / self.batch_size),
		
		v_acc = 0
		v_loss = 0
		m = 2
		for i in range(self.valid_num/m):	
			n1 = m*i*self.batch_size
			n2 = (m*i+1)*self.batch_size
			v_acc += self.sess.run(accuracy, feed_dict = {self.x_nn: self.hotx(self.x_valid[n1:n2]), 
				self.y_nn: self.hoty(self.y_valid[n1:n2]), self.bs: self.batch_size, self.keep_prob: 1}) 
			v_loss += self.sess.run(loss, feed_dict = {self.x_nn: self.hotx(self.x_valid[n1:n2]), 
				self.y_nn: self.hoty(self.y_valid[n1:n2]), self.bs: self.batch_size, self.keep_prob: 1}) 
			v_acc /= (self.valid_num/m)
		v_loss /= (self.valid_size/m)
		
		print ", v_acc: %.3f" % v_acc,
		print ", v_loss: %.4f" % v_loss

	def construct_id_w(self):
		common = pickle.load(open(self.path + "model/common1_word", "rb"))
		self.id_w = {}
		for i in range(self.vocab_size): self.id_w[i] = common[i]
		print len(self.id_w)
		for i in range(10): print self.id_w[i], 
		pickle.dump(self.id_w, open(self.path + "model/dict_id_w", "wb"))

	def get_ans_id(self, n):
		if 	 n == 0: return "a"
		elif n == 1: return "b"
		elif n == 2: return "c"
		elif n == 3: return "d"
		elif n == 4: return "e"
		else: return "error!"

	def output_answer(self, total_prob):
		fout = open(self.path + "/result/result2", "wb")
		fout.write("id,answer\n") 
		for i in range(self.y_test_id.shape[0]):
			n = total_prob[i].argmax(axis = 0)
			ans = self.get_ans_id(n)
			fout.write("%d,%s\n" %(i+1, ans)) 

	def load_test(self):
		print "Loading data"
		self.x_test = pickle.load(open(self.path + "model/x_test", "rb"))
		self.x_id = pickle.load(open(self.path + "model/x_test_id", "rb"))
		self.y_test = pickle.load(open(self.path + "model/y_test", "rb"))
		self.y_test_id = pickle.load(open(self.path + "model/y_test_id", "rb"))
		self.w_id = pickle.load(open(self.path + "model/dict", "rb"))
		# self.id_w = pickle.load(open(self.path + "model/dict_id_w", "rb"))

	def compute_answer(self, sess, pred):
		self.load_test()
		back = 2 # consider the prob of latter 2 words
		
		n = self.x_test.shape[0]
		prob = np.empty([n, 5, back+1], dtype = float)
		x_t1 = np.empty([1, 20, self.dim_size])
		x_t5 = np.empty([5, 20, self.dim_size])
		for i in range(n):
			x_t1[0] = self.x_test[i][:20] # for only one testing data
			y_t1_id = self.y_test_id[i]
			ans = sess.run(pred, feed_dict = {self.x_nn: x_t1, self.bs: 1, self.keep_prob: 1}) 
			for j in range(5): prob[i][j][0] = ans[0][y_t1_id[j]] # input 5 probable y id

			for k in range(back):
				for j in range(5):
					x_t5[j][:20-k] = np.concatenate((self.x_test[i][1+k:20], self.y_test[i][j:j+1]), axis = 0)
					if k != 0:
						x_t1[0] = np.concatenate((x_t5[j][:20-k], self.x_test[i][21:21+k]), axis = 0)
				ans = sess.run(pred, feed_dict = {self.x_nn: x_t5, self.bs: 5, self.keep_prob: 1}) 
				for j in range(5):
					prob[i][j][k+1] = ans[j][self.x_id[i][k]] # input real next x id

		total_prob = np.empty([n, 5])
		for i in range(n):
			for j in range(5):
				total_prob[i][j] = prob[i][j][0]+prob[i][j][1]+prob[i][j][2]

		self.output_answer(total_prob)

	def early_stop(self, new_v):
		if new_v < self.min_loss_v: self.min_loss_v = new_v
		if new_v > self.min_loss_v + 0.003: print "early stop!!"
			
	def nn(self, first_time):
		print "Constructing lstm-nn model"
		for iii in range(1):
			self.sess = tf.Session()
			pred = self.lstm_layer_multi(self.x_nn)
			loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y_nn, logits = pred)))
			#+ tf.nn.l2_loss(self.w1)*self.beta1 + tf.nn.l2_loss(self.w4)*self.beta2)
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
			optimizer = tf.train.AdamOptimizer(self.lr)
			train_step = optimizer.apply_gradients(zip(grads, tvars))
			correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y_nn, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

			init = tf.global_variables_initializer()
			saver = tf.train.Saver()
		
			if self.from_last == 1:
				saver.restore(self.sess, self.path + "/model/model.ckpt")
		  		print("Model restored.")
		  	else:
				self.sess.run(init)
				
			self.step = 0
			batch_id = 0
			
			print "\nStart training"
			flag = 1
			for i in range(1):
				while self.step < self.epoch:
					if self.small == 0:
						begin, end = self.cut_batch(self.step)
						x_batch = self.x[begin:end]
						y_batch = self.y_train[begin:end]
						
					self.sess.run([train_step], feed_dict = {self.x_nn: self.hotx(x_batch), self.y_nn: self.hoty(y_batch), 
						self.bs: self.batch_size, self.keep_prob: self.drop})
					
					if self.step % 150 == 0:
						print self.step, self.batch_id,
						self.output_accuracy(loss, accuracy)

					if self.step == 3150 or self.step == 6000:
						self.lr *= 0.1
						print "lr = ", self.lr

					if self.step == 3150:
						self.x, self.y_train = self.shuffle_in_unison(self.x, self.y_train)
					
					self.step += 1
				
				save_path = saver.save(self.sess, self.path + "/model/model.ckpt")
	  			print("Model saved in file: %s" % save_path)

def main():
	model = Model()
	model.set_nn_parameter()

	model.load_x_train()
	model.load_y_train()
	model.x, model.y_train = model.shuffle_in_unison(model.x, model.y_train)
	model.cut_valid()
	model.nn(1)
	
	print "\n Compile successfully!"

if __name__ == '__main__':
	main()


