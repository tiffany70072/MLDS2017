# final project: Adam
# experiment 3: housing price

from numpy import genfromtxt, array
from random import random
from math import *
import numpy as np
import pylab as plt

class Model(object):
	def __init__(self):
		self.feature = [3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15] 
		
	def process_raw_data(self):
		# import data from csv by numpy
		self.data = genfromtxt('kc_house_data.csv', delimiter = ',')
		print "raw_data.shape = ", self.data.shape
		
		self.data = np.delete(self.data, 0, axis = 0) # remove top 1 row
		self.normalization()
		# reshape data
		self.x = self.data[:, self.feature]
		self.y = self.data[:, 2]

	def get_training_data(self):
		self.x = array(self.x)
		self.y = array(self.y)
		
		print "x.shape = ", self.x.shape
		print "y.shape = ", self.y.shape
		print self.x[0]
		print self.y[0]
			
		np.save("x_train", self.x)
		np.save("y_train", self.y)

	def normalization(self):
		self.data[:, 2] = self.data[:, 2]/10000.0 	# y
		self.data[:, 3] = self.data[:, 3]*1.0		# 0
		self.data[:, 4] = self.data[:, 4]*1.0
		self.data[:, 5] = self.data[:, 5]/1000.0
		self.data[:, 6] = self.data[:, 6]/10000.0	# 3
		self.data[:, 8] = self.data[:, 8]*100.0		# 4
		self.data[:, 9] = self.data[:, 9]*10.0		# 5
		self.data[:, 10] = self.data[:, 10]*1.0
		self.data[:, 11] = self.data[:, 11]*1.0
		self.data[:, 12] = self.data[:, 12]/1000.0
		self.data[:, 13] = self.data[:, 13]/100.0
		self.data[:, 14] = 2017 - self.data[:, 14]	# 10
		#self.data[:, 15] = 2017 - self.data[:, 15]
		for i in range(self.data.shape[0]):
			if self.data[i, 15] == 0: self.data[i, 15] = self.data[i, 14]
			else: self.data[i, 15] = 2017 - self.data[i, 15]

		self.data[:, 3:16] = self.data[:, 3:16] * 0.1

	def compute_normalization(self):
		normal = np.mean(self.x, 0)
		for i in range(normal.shape[0]):
			print i, "%.3f" %normal[i]

model = Model()
model.process_raw_data()
model.get_training_data()
model.compute_normalization()

