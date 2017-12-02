# final project: Adam
# experiment 1: MNIST

from numpy import genfromtxt, array
from math import log
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

import tensorflow as tf
import numpy as np
import pickle
import sys
import os

os.environ["THEANO_FLAGS"] = "device=gpu0"
	
def load_test_data():
	print "Finished Training. Start Loading Testing Data"
	test = pickle.load(open(sys.argv[1]+"/test.p", "rb"))
	print "Finished Loading"

	test_raw = array(test.get("data"))
	temp = np.empty([3, 32, 32])
	x_test = np.empty([10000, 3, 32, 32])
	for i in range(10000):
		for j in range(3):
			for k in range(32):
				for l in range(32):
					temp[j][k][l] = test_raw[i][1024*j+32*k+l]
		x_test[i] = temp
	
	print "x_test.shape = ", x_test.shape
	return x_test

def output_result(result): # write file
	print result.shape
	fout = open(sys.argv[3], 'w')

	fout.write("ID,class\n")
	for i in range(10000):
		maxP = 0.0
		maxId = -1
		for j in range(10):
			if result[i][j] > maxP:
				maxP = result[i][j]
				maxId = j
		fout.write("%d,%d\n" % (i, maxId))

def test(model):
	x_test = load_test_data()
	result = model.predict(x_test)
	output_result(result)

model = load_model(sys.argv[2])
test(model)
