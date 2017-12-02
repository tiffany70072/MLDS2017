import tensorflow as tf
import numpy as np
#import skimage
#import skimage.io
#import skimage.transform
import scipy.misc
import my_library as my_lib
import random
import pickle

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config = config))

class Model(object):
	def __init__(self):
		self.first_train = 1
		#self.gan = "mine"
		self.gan = "example"
		self.gan = "wgan"
		#self.gan = 'wgan-gp'
		self.drop = 1
		if self.gan == "wgan": self.lr = 5e-5 
		else: self.lr = 2e-4
		self.lam = 0.1
		self.d_step, self.g_step = 10, 4
		self.beta = 0.5
		self.clip = 0.01
		self.lamb = 10.0

		self.hair_size = 12
		self.eyes_size = 10

		self.data_size = 10966
		self.batch_size = 64
		self.sample_size = 108
		self.train_size = 10880
		self.train_batch = self.train_size / self.batch_size # num of train batch
		self.valid_size = self.data_size - self.train_size # num of valid
		self.valid_batch = self.valid_size / self.batch_size
		self.total_epoch = int(self.train_batch * 200)
		
		self.text_dim = self.hair_size + self.eyes_size
		self.noise_dim = 100 # ??? 78?
		self.input_dim = self.text_dim + self.noise_dim
		self.pict_dim = 64
		self.image_shape = [self.pict_dim, self.pict_dim, 3]
		
		self.sess = tf.Session()

		self.name = "_wgan_adam"
		self.fout = open('terminal' + self.name, 'w')

	def load_pict(self):
		'''self.pict = []
		for i in range(len(self.tags)):
			idx = self.tags[i][0]
			img = skimage.io.imread("../data/faces/"+str(idx)+".jpg")
			img_resized = skimage.transform.resize(img, (64, 64))
			self.pict.append(img_resized)
		self.pict = np.array(self.pict)
		print img.shape, img_resized.shape
		np.save("../data/my_pict", self.pict)'''
		self.pict = np.load("../data/my_pict.npy")
		print self.pict.shape
	
	def load_tags(self):
		self.tags = np.load("../data/my_tags.npy")
		self.tags = my_lib.mul_hot(self.tags, self.data_size, self.hair_size, self.eyes_size)

	def load(self):
		self.load_tags()
		self.load_pict()

		self.tags_valid = self.tags[self.train_size:self.train_size + self.valid_size]
		self.pict_valid = self.pict[self.train_size:self.train_size + self.valid_size]
		self.tags = self.tags[:self.train_size]
		self.pict = self.pict[:self.train_size]

		print "tags_train =", self.tags.shape
		print "pict_train =", self.pict.shape#.shape
		print "tags_valid =", self.tags_valid.shape
		print "pict_valid =", self.pict_valid.shape

	def set_nn_variable(self):
		print "Setting Variable"
		self.right_text_nn = tf.placeholder(tf.float32, [None, self.text_dim], name = "text")
		self.wrong_text_nn = tf.placeholder(tf.float32, [None, self.text_dim], name = "wrong_text")
		self.noise_nn = tf.placeholder(tf.float32, [None, self.noise_dim])
		self.real_img_nn = tf.placeholder(tf.float32, [None] + self.image_shape, name = 'real_images')
		self.wrong_img_nn = tf.placeholder(tf.float32, [None] + self.image_shape, name = 'wrong_images')
		self.sample_img_nn = tf.placeholder(tf.float32, [None] + self.image_shape, name = 'sample_images')

		self.keep_prob = tf.placeholder("float")
		self.bs = tf.placeholder(dtype = tf.int32)

	def generator(self, z):
		print "generator"
		with tf.variable_scope("generator") as scope: # may be useless
			self.text_encoding = my_lib.matmul(self.right_text_nn, self.text_dim, 128, 'g_txt')
			x0 = tf.concat([z, self.text_encoding], axis = 1)
			print x0.shape

			x1 = tf.nn.relu(my_lib.matmul(x0, self.noise_dim + 128, self.pict_dim*16*4*4, 'g_0'))
			x1 = tf.reshape(x1, [-1, 4, 4, self.pict_dim * 16])
			x1 = tf.nn.relu(my_lib.batch_norm(x1, 'g_bn1'))
			print x1.shape
			x2 = my_lib.conv2d_transpose(x1, [5, 5, self.pict_dim*8, self.pict_dim*16], [self.batch_size, 8, 8, self.pict_dim*8], 'g_1')
			x2 = tf.nn.relu(my_lib.batch_norm(x2, 'g_bn2'))
			print x2.shape
			x3 = my_lib.conv2d_transpose(x2, [5, 5, self.pict_dim*4, self.pict_dim*8], [self.batch_size, 16, 16, self.pict_dim*4], 'g_2')
			x3 = tf.nn.relu(my_lib.batch_norm(x3, 'g_bn3'))
			print x3.shape
			x4 = my_lib.conv2d_transpose(x3, [5, 5, self.pict_dim*2, self.pict_dim*4], [self.batch_size, 32, 32, self.pict_dim*2], 'g_3')
			x4 = tf.nn.relu(my_lib.batch_norm(x4, 'g_bn4'))
			print x4.shape
			x5 = my_lib.conv2d_transpose(x4, [5, 5, 3, self.pict_dim*2], [self.batch_size, 64, 64, 3], 'g_4')
			print x5.shape

			return tf.nn.tanh(x5)

	def sampler(self, z):
		tf.get_variable_scope().reuse_variables()

	def discriminator(self, pict, text, reuse = False):
		print "discriminator"
		with tf.variable_scope("discriminator") as scope:
			if reuse: scope.reuse_variables()
			if reuse == False: print pict.shape

			x0 = my_lib.lrelu(my_lib.conv2d(pict, [5, 5, 3, self.pict_dim], 'd_0'))
			if reuse == False: print x0.shape
			x1 = my_lib.lrelu(my_lib.batch_norm(my_lib.conv2d(x0, [5, 5, self.pict_dim, self.pict_dim*2], 'd_1'), 'd_bn1'))
			if reuse == False: print x1.shape
			x2 = my_lib.lrelu(my_lib.batch_norm(my_lib.conv2d(x1, [5, 5, self.pict_dim*2, self.pict_dim*4], 'd_2'), 'd_bn2'))
			if reuse == False: print x2.shape
			x3 = my_lib.lrelu(my_lib.batch_norm(my_lib.conv2d(x2, [5, 5, self.pict_dim*4, self.pict_dim*8], 'd_3'), 'd_bn3'))
			if reuse == False: print x3.shape
			x3 = tf.reshape(x3, [-1, 8192])
			x3 = tf.concat([self.text_encoding, x3], axis = 1)
			if reuse == False: print x3.shape
			x4 = my_lib.matmul(x3, 8192 + 128, 1, 'd_4')
			if reuse == False: print x4.shape

			return tf.nn.sigmoid(x4), x4

	def pretrain_loss(self, samp, real):
		samp = tf.reshape(samp, [self.batch_size, self.pict_dim*self.pict_dim*3])
		real = tf.reshape(real, [self.batch_size, self.pict_dim*self.pict_dim*3])
		return tf.reduce_mean(tf.square(samp - real))#, samp, real

	def build_model(self):
		print "Constructing rnn model"
		self.G = self.generator(self.noise_nn)
		self.D_real, self.D_real_logits = self.discriminator(self.real_img_nn, self.right_text_nn) # sr, real
		self.D1, self.D1_logits = self.discriminator(self.G, self.right_text_nn, reuse = True) # sf, fake
		#self.D2, self.D2_logits = self.discriminator(self.real_img_nn, self.wrong_text_nn, reuse = True) # sw
		#self.D2, self.D2_logits = self.discriminator(self.G, self.wrong_text_nn, reuse = True) # sw
		self.D2, self.D2_logits = self.discriminator(self.wrong_img_nn, self.right_text_nn, reuse = True)

		if self.gan == "example":
			self.g_loss = tf.reduce_mean(\
				tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D1_logits, labels = tf.ones_like(self.D1)))
			self.d_loss_real = tf.reduce_mean(\
				tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_real_logits, labels = tf.ones_like(self.D_real)))
			self.d_loss_1 = tf.reduce_mean(\
				tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D1_logits, labels = tf.zeros_like(self.D1)))
			self.d_loss_2 = tf.reduce_mean(\
				tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D2_logits, labels = tf.zeros_like(self.D2)))
			#self.d_loss_3 = tf.reduce_mean(\
			#	tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D3_logits, labels = tf.zeros_like(self.D3)))
			#self.d_loss_fake = self.d_loss_1 + self.d_loss_2 + self.d_loss_3

		elif self.gan == "mine":
			self.g_loss = tf.reduce_mean(tf.log(self.D1))
			self.d_loss_real = tf.reduce_mean(tf.log(self.D_real))
			self.d_loss_1 = tf.reduce_mean(tf.log(1 - self.D1))
			self.d_loss_2 = tf.reduce_mean(tf.log(1 - self.D2))
			self.d_loss = self.d_loss_real + (self.d_loss_1 + self.d_loss_2)/2.0

		elif self.gan == "wgan":
			self.g_loss = tf.reduce_mean(- self.D1_logits)
			self.d_loss_real = tf.reduce_mean(self.D_real_logits)
			self.d_loss_1 = tf.reduce_mean(- self.D1_logits)
			#self.d_loss_2 = tf.reduce_mean(- self.D2_logits)
			#self.d_loss_3 = tf.reduce_mean(- self.D3_logits)
			#self.d_loss = self.d_loss_real + (self.d_loss_1 + self.d_loss_2)/2.0
			self.d_loss = self.d_loss_real + self.d_loss_1

		elif self.gan == "wgan-gp":
			self.g_loss = tf.reduce_mean(- self.D1_logits)
			self.d_loss_real = tf.reduce_mean(- self.D_real_logits) # fake - real
			self.d_loss_1 = tf.reduce_mean(self.D1_logits)
			self.d_loss_2 = tf.reduce_mean(self.D2_logits)

			alpha = tf.random_uniform(shape = [self.batch_size, 1], minval = 0., maxval = 1.)
			differences = self.G - self.real_img_nn # fake_data - real_data
			interpolates = self.real_img_nn + (alpha*differences)
			temp = self.discriminator(interpolates, self.right_text_nn, reuse = True)
			gradients = tf.gradients(temp[1], [interpolates])[0] 
			#print "ya"
			slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices = [1])) # ??
			gradient_penalty = tf.reduce_mean((slopes-1.)**2)
			
			self.d_loss = self.d_loss_real + (self.d_loss_1 + self.d_loss_2)/2.0 \
		    	+ self.lamb*gradient_penalty 

		#self.pre_loss = self.pretrain_loss(self.G, self.real_img_nn)
		# + self.d_loss_3)/3.0

		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if 'd_' in var.name]		
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

	def next_batch(self, step, isTrain = True, img = False):
		if isTrain == True:
			n1 = step%self.train_batch
			n2 = (step + random.randint(1, self.train_batch-1))%self.train_batch
			begin1, end1 = self.batch_size*n1, self.batch_size*(n1+1)
			begin2, end2 = self.batch_size*n2, self.batch_size*(n2+1)
			self.real_img_batch = self.pict[begin1:end1]
			self.right_txt_batch = self.tags[begin1:end1]
			#self.wrong_txt_batch = (self.tags[begin1:end1] + self.tags[begin2:end2])/2.0
			self.wrong_img_batch = self.pict[begin2:end2]
			#if img == False: self.wrong_img_batch = self.pict[begin2:end2]
			
		else:
			n = step%self.valid_batch
			begin, end = self.batch_size*n, self.batch_size*(n+1)
			self.real_img_batch = self.pict_valid[begin:end]
			self.text_batch = self.tags_valid[begin:end]
		self.noise_batch = np.random.normal(0, 1, [self.batch_size, self.noise_dim]).astype(np.float32)
		#self.noise_batch = np.random.uniform(-1, 1, [self.batch_size, self.noise_dim]).astype(np.float32)

	def output_d_loss(self):
		print "d_loss =",
		d_loss = 0
		for i in range(10):
			self.next_batch(i*15, True)
			d_loss += self.sess.run(self.d_loss, feed_dict = {self.noise_nn: self.noise_batch, \
					self.real_img_nn: self.real_img_batch, self.right_text_nn: self.right_txt_batch, \
					self.wrong_img_nn: self.wrong_img_batch}) #self.wrong_text_nn: self.wrong_txt_batch, 
		print d_loss/10.0
		self.fout.write("d_loss = " + str(d_loss) + "\n")

	def output_g_loss(self):
		print "-- g_loss =",
		g_loss = 0
		for i in range(10):
			self.next_batch(i*15, True)
			g_loss += self.sess.run(self.g_loss, feed_dict = {\
				self.real_img_nn: self.real_img_batch, self.noise_nn: self.noise_batch, self.right_text_nn: self.right_txt_batch})
		print g_loss/10.0
		self.fout.write("-- g_loss = " + str(g_loss) + "\n")

	def save_images(self, name):
	 	self.next_batch(step = 0, img = True)
		fake_img = self.sess.run(self.G, feed_dict = {self.noise_nn: self.noise_batch, self.right_text_nn: self.right_txt_batch})
	 	for i in range(6): self.save_one_image(fake_img[i], i, name)

	def save_one_image(self, img, idx, name):
		scipy.misc.imsave('../fake_img/img' + str(name) + "-" + str(idx) + '.jpg', img)

	def set_opt(self):
		if self.gan != "wgan":
			self.d_optim = tf.train.AdamOptimizer(self.lr, beta1 = self.beta).minimize(self.d_loss, var_list = self.d_vars)
			self.g_optim = tf.train.AdamOptimizer(self.lr, beta1 = self.beta).minimize(self.g_loss, var_list = self.g_vars)
			# can change beta1
		elif self.gan == "wgan":
			self.d_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_vars]
			#self.d_optim = tf.train.RMSPropOptimizer(10e-5, name = 'd_optim').minimize(- self.d_loss, var_list = self.d_vars)
			#self.g_optim = tf.train.RMSPropOptimizer(10e-5, name = 'g_optim').minimize(self.g_loss, var_list = self.g_vars)
			self.d_optim = tf.train.AdamOptimizer(10e-5, name = 'd_optim').minimize(- self.d_loss, var_list = self.d_vars)
			self.g_optim = tf.train.AdamOptimizer(10e-5, name = 'g_optim').minimize(self.g_loss, var_list = self.g_vars)
		elif self.gan == "wgan-gp":
			self.g_optim = tf.train.AdamOptimizer(10e-5, beta1 = 0.5, beta2 = 0.9).minimize(self.g_loss, var_list = self.g_vars)
			self.d_optim = tf.train.AdamOptimizer(10e-5, beta1 = 0.5, beta2 = 0.9).minimize(self.d_loss, var_list = self.d_vars)
	
		#self.pre_optim = tf.train.AdamOptimizer(1e-3).minimize(self.pre_loss, var_list = self.g_vars)

	def pretrain(self):
		print "Pretrain"
		for i in range(750):
			self.next_batch(step = 0)
			if i % 249 == 0:
				print "Epoch =", i
				
				loss = 0
				for j in range(10):
					self.next_batch(j*15, True)
					loss += self.sess.run(self.pre_loss,\
						feed_dict = {self.noise_nn: self.noise_batch, \
						self.real_img_nn: self.real_img_batch, self.right_text_nn: self.right_txt_batch})
				print "pre_train_loss =", loss/10.0
			if i % 250 == 0: self.save_images("pretrain" + str(i))
			print "lala"
			self.next_batch(step = i)
			_ = self.sess.run([self.pre_optim],\
				feed_dict = {self.noise_nn: self.noise_batch, \
				self.real_img_nn: self.real_img_batch, self.right_text_nn: self.right_txt_batch})
		self.save_images("pretrain" + str(i))
		print "Save!"

	def train(self):	
		self.saver = tf.train.Saver()
		init = tf.global_variables_initializer()

		if self.first_train == 0:
			ckpt = tf.train.get_checkpoint_state("../model")
			#self.saver.restore(self.sess, "../model/model" + self.name + ".ckpt")
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
	 		print("Model restored.")
	  	else: self.sess.run(init)
		
		print "Start Training"
		print "Model =", self.gan

		#self.pretrain()

		for self.epoch in range(self.total_epoch):
			if self.epoch%200 == 0: 
				print "\nEpoch =", self.epoch, self.epoch/self.train_batch
				self.fout.write("\nEpoch = " + str(self.epoch) + ", " + str(self.epoch/self.batch_size) + "\n")
			if self.epoch%250 == 0: self.save_images(str(self.epoch + 0*100))
			
			self.next_batch(step = self.epoch)

			# update D
			if self.gan == "example" or self.gan == "mine":
				_ = self.sess.run([self.d_optim],\
					feed_dict = {self.noise_nn: self.noise_batch, \
					self.real_img_nn: self.real_img_batch, self.right_text_nn: self.right_txt_batch, \
					self.wrong_text_nn: self.wrong_txt_batch, })
					#self.wrong_img_nn: self.wrong_img_batch, })
				if self.epoch % 200 < 10: self.output_d_loss()
			elif self.gan == "wgan":
				if self.epoch%4 < 4 or self.epoch < 20:
					_, _ = self.sess.run([self.d_optim, self.d_clip], \
					 	feed_dict = {self.noise_nn: self.noise_batch, \
						self.real_img_nn: self.real_img_batch, self.right_text_nn: self.right_txt_batch}) \
						#self.wrong_img_nn: self.wrong_img_batch, self.wrong_text_nn: self.wrong_txt_batch})
					if self.epoch % 300 < 10: self.output_d_loss()
			elif self.gan == "wgan-gp":
				if True: #self.epoch < 20 or 
					_ = self.sess.run(self.d_optim, \
					 	feed_dict = {self.noise_nn: self.noise_batch, \
						self.real_img_nn: self.real_img_batch, self.right_text_nn: self.right_txt_batch, \
						self.wrong_img_nn: self.wrong_img_batch})
					if self.epoch % 200 < 20: self.output_d_loss()
		
			# update G 
			if (self.gan == "example" or self.gan == "mine") and self.epoch > 20:
				for i in range(2):
					_ = self.sess.run([self.g_optim],\
						feed_dict = {self.noise_nn: self.noise_batch, self.right_text_nn: self.right_txt_batch})
					if self.epoch % 200 < 10: self.output_g_loss()
			if self.gan == "wgan" or self.gan == "wgan-gp":
				if self.epoch%2 < 1 and self.epoch > 20: #  and 
					_ = self.sess.run(self.g_optim,\
						feed_dict = {self.noise_nn: self.noise_batch, self.right_text_nn: self.right_txt_batch})
					if self.epoch % 200 < 20: self.output_g_loss()
		
			if self.epoch % 1000 == 0 and self.epoch != 0: self.save_model()
		self.save_model()

	def save_model(self):
		save_path = self.saver.save(self.sess, "../model/model" + self.name + ".ckpt", global_step = self.epoch)
		print "Model saved in file: %s" % save_path

	def testing(self):
		f = open("testing_text", "r")
		path = "../data/"
		hair_dict = pickle.load(open(path + "hair_dict", "rb"))
		eyes_dict = pickle.load(open(path + "eyes_dict", "rb"))
		test = []
		for i, line in enumerate(f):
			temp = np.zeros([22])
			l = line.split(",")
			l = l[1].split()
			try:
				temp[hair_dict[l[0]]] = 1
				print hair_dict[l[0]],
			except KeyError: print i, l[0]
			try:
				temp[eyes_dict[l[2]]+12] = 1
				print eyes_dict[l[2]]+12
			except KeyError: print i, l[2]
			
			test.append(temp)
		test = np.array(test)
		print test.shape
		self.right_txt_batch = self.tags[0:self.batch_size]

		self.saver = tf.train.Saver()
		init = tf.global_variables_initializer()

		ckpt = tf.train.get_checkpoint_state("../model-gp-0523-w-img")	
		self.saver.restore(self.sess, ckpt.model_checkpoint_path)
	 	print("Model restored.\nTesting")
		
		for i in range(test.shape[0]):
			self.noise_batch = np.random.normal(0, 1, [self.batch_size, self.noise_dim]).astype(np.float32)
			
			for j in range(5): self.right_txt_batch[:5] = test[i]
			img = self.sess.run(self.G, feed_dict = {self.noise_nn: self.noise_batch, self.right_text_nn: self.right_txt_batch})
	 		for j in range(5): 
	 			scipy.misc.imsave('../../sample/gp2/sample_' + str(i+1) + "_" + str(j+1) + '.jpg', img[j])

def main():
	model = Model()

	model.load()
	model.set_nn_variable()
	model.build_model()
	model.set_opt()
	model.train()
	for i in range(3): model.save_images("final" + str(i))
	#model.testing()

	print "Compile Successfully!"

if __name__ == '__main__':
	main()

