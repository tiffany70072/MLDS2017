'''
HW0 in MLDS
Number recognition for MNIST
'''

from numpy import genfromtxt, array
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, \
						Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

import numpy as np
import pickle
import struct
import sys
import gzip


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config = config))

class NumberRecog(object):
	def __init__(self):
		self.const1 = 1

	def load_train_x_data(self):
		print "Loading Labeled Data"
		# open binary file
		bin_data = open("data/train-image", 'rb').read()
		offset = 0
		fmt_header = '>iiii'
		magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
		
		# resize
		image_size = num_rows * num_cols
		offset += struct.calcsize(fmt_header)
		fmt_image = '>' + str(image_size) + 'B'
		self.train_x = np.empty((num_images, 1, num_rows, num_cols))
		for i in range(num_images):
			# if (i + 1) % 10000 == 0: print 'Finished %d' % (i + 1) + 'pages'
			self.train_x[i][0] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
			offset += struct.calcsize(fmt_image)
		print "train_x = ", self.train_x.shape

	def load_train_y_data(self):
		bin_data = open("data/train-label", 'rb').read()

		# open binary file
		offset = 0
		fmt_header = '>ii'
		magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
		
		# resize
		offset += struct.calcsize(fmt_header)
		fmt_image = '>B'
		self.train_y = np.empty(num_images, dtype = int)
		for i in range(num_images):
			self.train_y[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
			offset += struct.calcsize(fmt_image)
		print "train_y = ", self.train_y.shape
		

	def hot_vector(self):
		self.train_y = np.zeros([60000, 10], dtype = int)
		for i in range(60000):		
			self.train_y[i][self.train_y[i]] = 1

	def load_test_x_data(self):
		bin_data = open("data/test-image", 'rb').read()
		offset = 0
		fmt_header = '>iiii'
		magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
		
		# resize
		image_size = num_rows * num_cols
		offset += struct.calcsize(fmt_header)
		fmt_image = '>' + str(image_size) + 'B'
		self.test_x = np.empty((num_images, 1, num_rows, num_cols))
		for i in range(num_images):
			# if (i + 1) % 10000 == 0: print 'Finished %d' % (i + 1) + 'pages'
			self.test_x[i][0] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
			offset += struct.calcsize(fmt_image)
		print "test_x = ", self.test_x.shape

	def load_data(self):
		self.load_train_x_data()
		self.load_train_y_data()
		self.hot_vector() # change train_y into hot vector
		print "Finished Data Preprocessing"

	def training(self):
		model = Sequential()
		model.add(Convolution2D(30, 3, 3, dim_ordering = "th", input_shape = (1, 28, 28)))
		model.add(MaxPooling2D((2, 2),    dim_ordering = "th"))
		model.add(Convolution2D(60, 3, 3, dim_ordering = "th"))
		model.add(MaxPooling2D((2, 2),    dim_ordering = "th"))
		model.add(Flatten())
		model.add(Dense(output_dim = 1200, W_regularizer = l2(0)))
		#  model.add(Activation('relu'))
		model.add(Activation('sigmoid'))
		model.add(Dropout(0.5))
		model.add(Dense(output_dim = 10))
		# model.add(Activation('softmax'))

		model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
			
		model.summary()
		model.fit(self.train_x, self.train_y, batch_size = 1000, nb_epoch = 2, validation_split = 0.1)
		model.save('model' + str(n) + '.h5')  # creates a HDF5 file 'model.h5'
		# del model  # deletes the existing model
		
		'''
		score = model.evaluate(self.valid_x, self.valid_y)
		print "\nTotal loss on CV set: ", score[0]
		print "Accuracy on CV set: ", score[1]
		'''
		
		self.load_test_x_data()
		self.result = model.predict(self.test_x, batch_size = 10000)
		self.output_result()

	def output_result(self): # write file
		fout = open("result.csv", 'w')
		
		fout.write("id,label\n")
		for i in range(10000):
			maxP  = 0.0
			maxId = -1
			for j in range(10):
				if self.result[i][j] > maxP:
					maxP  = self.result[i][j]
					maxId = j
			fout.write("%d,%d\n" % (i, maxId))

task = NumberRecog()
task.load_data()
task.training()

"""
def cv_error(model, x_valid, y_valid):
	score = model.evaluate(x_valid, y_valid)
	print "\nTotal loss on CV set: ", score[0]
	print "Accuracy on CV set: ", score[1]

	x_test = load_test_data()
	result = model.predict(x_test)
	output_result(result)

x_label, y_label = load_label_data()
"""
