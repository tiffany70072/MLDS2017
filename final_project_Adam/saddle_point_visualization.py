# final project: Adam
# visualize how weight moves near saddle point

#from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def init(s):
	if s == 1:   return -3.0, 0.8
	elif s == 2: return -2.5, -0.1 #return -2.5, 1.5	

def func(x, y):
	#f1 = 0.5*x**2 - 0.1*x - 0.3*(y-0.4)**2 + 0.1*(y+1)**2 - 0.1
	#f2 = 4*x**2 + 2*y**2 - 2.5*(y-1)**2
	#f2 = 4*x**2 - 2*y**2 + 0.1*y - 4*(x-1)**2 - 2*y**2 + 0.1*y + 4*(x-1.5)**2 - 2*y**2 + 0.1*y
	f2 = (x-1.0)**2 - (y-1.0)**2
	return f2

def set_figure():
	l = 4.0
	fig = plt.figure(figsize=(10, 6), edgecolor = 'k')

	p = fig.add_subplot(111, projection = '3d')
	p.axis([-l, l, -l, l])
	if focus == True: p.axis([-3.5, -2, -1, 0.5])
	p.set_xlabel('W1')
	p.set_ylabel('W2')
	p.set_zlabel('Loss')

	x = y = np.arange(-l, l, 0.1)
	X, Y = np.meshgrid(x, y)
	zs = np.array([func(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z = zs.reshape(X.shape)

	surf = p.plot_surface(X, Y, Z, alpha = 0.3, cmap = cm.coolwarm, linewidth = 0, antialiased = False)

	# Customize the z axis.
	p.set_zlim(-30, 30)
	p.zaxis.set_major_locator(LinearLocator(10))
	p.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink = 0.5, aspect = 5)

	return p

def plot3d(p, a1, a2, a3, s, c):
	p.scatter(a1, a2, a3, c = c, marker = 'o', s = 5)
	return p

def train(opt_name):
	p1, p2, p3 = [], [], []

	w1 = tf.Variable(tf.constant(s[0], shape = [1]))
	w2 = tf.Variable(tf.constant(s[1], shape = [1]))
	loss = func(w1, w2)

	if opt_name == "adam1":
		opt = tf.train.AdamOptimizer(0.005, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8).minimize(loss)
	elif opt_name == "adam2":
		opt = tf.train.AdamOptimizer(0.005, beta1 = 0.1, beta2 = 0.999, epsilon = 1e-8).minimize(loss)
	elif opt_name == "adam3":
		opt = tf.train.AdamOptimizer(0.005, beta1 = 0.5, beta2 = 0.999, epsilon = 1e-8).minimize(loss)
	elif opt_name == "adam4":
		opt = tf.train.AdamOptimizer(0.005, beta1 = 0.999, beta2 = 0.999, epsilon = 1e-8).minimize(loss)
		#opt = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	print opt_name
	for i in range(600):
		if i%10 == 0:
			t1 = sess.run(w1)[0]
			t2 = sess.run(w2)[0]
			t3 = sess.run(loss)[0]
			print "%.4f, %.4f" %(t1, t2),
			print "%.4f" %t3

			p1.append(t1)
			p2.append(t2)
			p3.append(t3)
		sess.run(opt)

	print sess.run(loss)[0]
	return p1, p2, p3

def loss_fig(l, c, title):
	label = ["beta1 = 0.9", "beta1 = 0.1", "beta1 = 0.5", "beta1 = 0.999"]
	label = ["epsilon = 1e-8", "epsilon = 0.1", "epsilon = 1.0", "epsilon = 0.0"]
	label = ["beta2 = 0.999", "beta2 = 0.9", "beta2 = 0.0", "beta2 = 1.0"]
	label = ["beta1 = 0.9", "beta1 = 1.0", "beta1 = 0.5", "beta1 = 0.0"]
	n = 40
	x = []
	for i in range(60):
		x.append(i*10)
	for i in range(4):
		plt.plot(x, l[i], c = c[i])
		#plt.plot([i for i in range(60)], l2, c = c[1])
		#plt.plot([i for i in range(60)], l3, c = c[2])
		
		if i == 0:
			plt.annotate(label[i], xy = (n*10, l[i][n]), xytext = (n*10, l[i][n]+1), color = c[i])
		elif i == 1:
			plt.annotate(label[i], xy = (n*10, l[i][n]), xytext = (n*10-100, l[i][n]-5), color = c[i])
		elif i == 2:
			plt.annotate(label[i], xy = (n*10, l[i][n]), xytext = (n*10-100, l[i][n]-2.5), color = c[i])
		else:
			plt.annotate(label[i], xy = (n*10, l[i][n]), xytext = (n*10, l[i][n]+1), color = c[i])
	
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(title)

	plt.show()

s = init(1)
mode  = 2
focus = 0 
color = ['#1B2021', '#1A366B', '#028090', '#9B1D20']

if mode == 1: p = set_figure()
a1, a2, a3 = train("adam1")
if mode == 1: p = plot3d(p, a1, a2, a3, "adam1", color[0])
else: b1 = a3
a1, a2, a3 = train("adam2")
if mode == 1: pp = plot3d(p, a1, a2, a3, "adam2", color[1])
else: b2 = a3
a1, a2, a3 = train("adam3")
if mode == 1: pp = plot3d(p, a1, a2, a3, "adam3", color[2])
else: b3 = a3
a1, a2, a3 = train("adam4")
if mode == 1: pp = plot3d(p, a1, a2, a3, "adam4", color[3])
else: b4 = a3

#loss_fig(a3, color[0], "Adam: epsilon = 1e-8")
#loss_fig(a3, color[1], "Adam: epsilon = 0.1")
if mode != 1: loss_fig([b1, b2, b3, b4], color, "Adam: beta1 = variable; beta2, epsilon = default")
plt.show()

