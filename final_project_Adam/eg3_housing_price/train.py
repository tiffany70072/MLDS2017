# final project: Adam
# experiment 3: housing price

import numpy as np
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, normalization
from keras import optimizers

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config = config))

class Model(object):
	def __init__(self):
		self.x = np.load("x_train.npy")
		self.y = np.load("y_train.npy")

	def select_training_data(self):
		self.x_train, self.y_train = np.array(self.x), np.array(self.y)
		self.add_x()

		n = 18000
		self.x_valid = self.x[n:]
		self.y_valid = self.y[n:]
		self.x = self.x[:n]
		self.y = self.y[:n]
		
		print "x_train =", self.x.shape, self.x_valid.shape 
		print "y_train =", self.y.shape, self.y_valid.shape

	def add_x(self):
		grade0 = [0, 1, 6, 7]
		grade1 = [2, 3, 8, 9, 10]
		
		square = np.square(self.x[:, grade0])
		sqrt = np.sqrt(self.x[:, :])#grade1])

		n = self.x_train.shape[1]
		cross_term = np.empty([self.x.shape[0], n*(n-1)/2])
		cube = np.power(self.x[:, grade0], 3)
		
		s = 0
		for i in range(n-1):
			for j in range(i+1, n):
				cross_term[:, s] = self.x[:, i] * self.x[:, j]
				s += 1

		self.x = np.concatenate([self.x, square], 1)
		self.x = np.concatenate([self.x, cross_term], 1)
		self.x = np.concatenate([self.x, np.ones([self.x.shape[0], 1])], 1)
		
		print self.x.shape
		
	def train(self):
		nn_model = Sequential()

		#nn_model.add(normalization.BatchNormalization(input_shape = self.x_train.shape))
		#nn_model.add(Dense(output_dim = 1200))

		nn_model.add(Dense(input_dim = self.x.shape[1], output_dim = 1000))
		nn_model.add(Activation('relu'))
		#nn_model.add(Dropout(0.4))
		nn_model.add(Dense(output_dim = 1000))
		nn_model.add(Activation('relu'))
		#nn_model.add(Dropout(0.4))
		nn_model.add(Dense(output_dim = 1000))
		#nn_model.add(Activation('sigmoid'))
		#model.add(Dropout(0.25))
		nn_model.add(Dense(output_dim = 1))
		#model.add(Activation('softmax'))
		nn_model.summary()
		
		opt = optimizers.Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, decay=0.0)
		#opt = optimizers.SGD(lr = 10E-5)
		nn_model.compile(loss = 'mean_squared_error', optimizer = opt, metrics = ['accuracy'])
		nn_model.fit(self.x, self.y, batch_size = 100, nb_epoch = 0, shuffle = True, validation_data = (self.x_valid, self.y_valid))

		nn_model.save('model.h5')  
		
		fout = open("result", 'w')
		self.result = nn_model.predict(self.x[:5000])
		self.output_result(fout, self.y[:5000])
		self.result = nn_model.predict(self.x_valid)
		self.output_result(fout, self.y_valid)

	def output_result(self, fout, y_true): # write file
		fout.write("y_pred, y_train, error, rms_error\n")
		ave_error = 0
		rms_error = 0
		count = self.result.shape[0]
		for i in range(self.result.shape[0]):
			if self.y[i] > 0:
				err1 = np.abs((self.result[i][0] - y_true[i]))/y_true[i]#self.y[i][0]
				ave_error += err1
				err2 = np.square((self.result[i][0] - y_true[i]))
				rms_error += err2
				
				fout.write("%.2f" %(self.result[i][0]) + " - " + "%.2f" %(y_true[i]) + " - ")
				fout.write("%.2f" %(err1*100) + ", %.2f" %(err2) + "\n")
			else:
				count -= 1

		ave_error = ave_error / float(count)
		rms_error = sqrt(rms_error / float(count))
		print "Number =", count
		print "Ave error = %.3f" %(ave_error * 100), "%"
		print "RMS error = %.3f" %(rms_error)


model = Model()
model.select_training_data()
model.train()
