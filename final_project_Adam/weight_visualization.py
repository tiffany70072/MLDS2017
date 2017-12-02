# final project: Adam
# visualize how weight moves during training

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt

data = np.load("b2_0.9_w.npy")
loss = np.load("b2_0.9_loss.npy")
data = np.load("1e-1_bad_w.npy")
loss = np.load("1e-1_bad_loss.npy")
print "data.shape = ", data.shape

w1 = data[:, 0]
w2 = data[:, 1]
w3 = data[:, 2]
t = [i for i in range(data.shape[0])]

'''
f, arr = plt.subplots(1, 3, sharey = True, figsize=(10, 4))
arr[0].plot(t, w1)
arr[0].set_title('w1')
arr[1].plot(t, w2)
arr[1].set_title('w2')
arr[2].plot(t, w3)
arr[2].set_title('w3')
plt.show()
#plt.title("Weights of good case on cifar-10")

#a2 = plt.figure()
#a2.plot(t, loss)
#a2.title = "loss"

plt.plot(t, loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
#plt.title('Sunday, 0918~0423, station = ' + str(i))
plt.show()'''

fig = plt.figure(figsize=(7, 5), edgecolor = 'k')

p = fig.add_subplot(111, projection = '3d')
#p.axis([-l, l, -l, l])
#if focus == True: p.axis([-3.5, -2, -1, 0.5])
p.set_xlabel('Parameter 1')
p.set_ylabel('Parameter 2')
p.set_zlabel('Loss')
plt.title('1e-1')

p.plot(w1, w2, loss, '-', c = 'b')
plt.show()

n = raw_data.shape[0]
n = 30
m = 100
m = 0

c = [(0.2, 0.7, 0.2), (1, 0.3, 0.1), (1, 0.8, 0.2), (0.1, 0.7, 0.7), (0.1, 0.2, 0.4)]
for i in range(5):
	x = raw_data[m:m+n, 2*i+0].tolist()
	y = raw_data[m:m+n, 2*i+1].tolist()

	plt.plot(x, y, "-", alpha = 0.5, c = (0.4, 0.5, 0.4))
	plt.plot(x, y, ".", c = c[i])
	#for i in range(len(x)):
	#	if n <= 50 or i % 10 == 9 or i == 0:
	#		plt.annotate(str(i+1), xy = (x[i], y[i]), xytext = (x[i]+0.01, y[i]+0.1)),
	            #arrowprops=dict(facecolor='black', shrink=0.005))

plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.title("Motion of 5 particles of Fe2O3")
plt.show()
