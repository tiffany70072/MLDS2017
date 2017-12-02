# final project: Adam
# experiment 1: MNIST

from numpy import genfromtxt, array
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam

import numpy as np
import sys
import pickle
import gzip
import struct

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config = config))

path = "../../NumberRecog/data/"

class NumberRecog(object):
	def __init__(self):
		self.optimizer = "adam"

	def load_train_x_data(self):
		print "Loading Labeled Data"
		# open binary file
		bin_data = open(path + "train-image", 'rb').read()
		offset = 0
		fmt_header = '>iiii'
		magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
		
		# resize
		image_size = num_rows * num_cols
		offset += struct.calcsize(fmt_header)
		fmt_image = '>' + str(image_size) + 'B'
		self.train_x = np.empty((num_images, 1, num_rows, num_cols))
		for i in range(num_images):
			#if (i + 1) % 10000 == 0: print 'Finished %d' % (i + 1) + 'pages'
			self.train_x[i][0] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
			offset += struct.calcsize(fmt_image)
		print "train_x = ", self.train_x.shape

	def load_train_y_data(self):
		bin_data = open(path + "train-label", 'rb').read()

		# open binary file
		offset = 0
		fmt_header = '>ii'
		magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
		
		#resize
		offset += struct.calcsize(fmt_header)
		fmt_image = '>B'
		self.train_y = np.empty(num_images, dtype = int)
		for i in range(num_images):
			self.train_y[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
			offset += struct.calcsize(fmt_image)
		print "train_y = ", self.train_y.shape

	def hot_vector(self):
		temp = np.zeros([60000, 10], dtype = int)
		for i in range(60000): temp[i][self.train_y[i]] = 1
		self.train_y = temp
		
	def load_test_x_data(self):
		bin_data = open(path + "test-image", 'rb').read()
		offset = 0
		fmt_header = '>iiii'
		magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
		#print 'magic: %d, num: %d, size: %d*%d' % (magic_number, num_images, num_rows, num_cols)
		
		# resize
		image_size = num_rows * num_cols
		offset += struct.calcsize(fmt_header)
		fmt_image = '>' + str(image_size) + 'B'
		self.test_x = np.empty((num_images, 1, num_rows, num_cols))
		for i in range(num_images):
			#if (i + 1) % 10000 == 0: print 'Finished %d' % (i + 1) + 'pages'
			self.test_x[i][0] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
			offset += struct.calcsize(fmt_image)
		print "test_x = ", self.test_x.shape

	def load_data(self):
		self.load_train_x_data()
		self.load_train_y_data()
		self.hot_vector() # change train_y into hot vector
		'''
		self.valid_x = self.train_x[55001:60000]
		self.valid_y = self.train_y[55001:60000]
		m = 0
		n = 55000
		self.train_x = self.train_x[m:n]
		self.train_y = self.train_y[m:n]
		'''
		print "Finished Data Preprocessing"

	def training(self):
		n = 30
		
		model = Sequential()
		model.add(Convolution2D(n, 3, 3, dim_ordering = "th", input_shape = (1, 28, 28)))
		model.add(MaxPooling2D((2, 2),    dim_ordering = "th"))
		model.add(Convolution2D(2*n, 3, 3, dim_ordering = "th"))
		model.add(MaxPooling2D((2, 2),    dim_ordering = "th"))
		model.add(Flatten())
		model.add(Dense(output_dim = n*20, W_regularizer = l2(0)))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(output_dim = 10))
		model.add(Activation('softmax'))

		self.optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 0, decay=0.0)
		model.compile(loss = 'categorical_crossentropy', optimizer = self.optimizer, metrics = ['accuracy'])

		model.summary()
		model.fit(self.train_x, self.train_y, batch_size = 1000, nb_epoch = 30, validation_split = 0.1)
		
	def output_result(self): # write file
		print self.result.shape
		fout = open("result-" + self.optimizer + ".csv", 'w')
		
def main():
	task = NumberRecog()
	task.load_data()
	task.training()

if __name__ == '__main__':
	main()
