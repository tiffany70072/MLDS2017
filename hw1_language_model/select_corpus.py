# Modify the corpus

from gensim.models import Word2Vec
import pickle
import numpy as np

class Model(object):
	def __init__(self):
		self.data_size 	= 3591 * 260
		self.batch_size = 200
		self.num_steps 	= 30 	# #(neurons) of input layer 1
		self.dim_size 	= 300 	# #(neurons) of input layer 2
		self.num_layers 	= 2 
		self.hidden_size = 1000 # unknown, 200, 650, 1500
		self.vocab_size = 9779	# #(neurons) of output layer (class)
		self.is_train 	= 16000
						
		self.path = "../"
		self.each_num = 260
		self.output_data_size = 49000
		#self.epoch_size = self.data_size / self.batch_size + 1
		# total = 3600
		# 130 - 391038 - 1100
		# 150 - 436000 - 1300
		# 170 - 480780 - 1400
		# 190 - 520000 - 1500
		# 260 - 636000 - 1900 -> (636000-1700*260)/1900 = 102

	def load(self):
		print "Loading"
		self.x = pickle.load(open("../model/freq_corpus_500", "rb"))
		#self.word_list = pickle.load(open("../model/word_list", "rb"))
		self.dict = pickle.load(open("../model/word_id", "rb"))
		
		'''print len(self.x) # 1830334
		print self.x[0]
		print len(self.word_list) # 11320 --> 11791
		print self.word_list[0:5]
		'''
		
		'''self.choice = pickle.load(open(self.path + "model/y_test_word", "rb"))
		self.dict = pickle.load(open(self.path + "model/dict", "rb"))

		self.y = []
		for i in range(1040):
			for j in range(5):
				w = self.choice[i][j]
				w = w.lower()
				w = w.replace("-", "")
				if w in self.dict and w not in self.y: 
					self.y.append(w)
					#print "1", w
				#elif w not in self.y: print "0", w
		print "num dict = ", len(self.y)
		pickle.dump(self.y, open(self.path + "model/choice_list", "wb"))

		'''
		self.choice = pickle.load(open(self.path + "model/choice_list", "rb"))
		#self.choice.remove("800")
		#self.choice.remove("1")
		#list of length = 3591
		#self.build_my_wordlist()
	
	def build_my_wordlist(self):
		self.word_id = {}
		count = 0
		for i in range(len(self.x)):
			for j in range(len(self.x[i])):
				if self.x[i][j] not in self.word_id:
					self.word_id[self.x[i][j]] = count
					count += 1
			if i % 100000 == 0: print i
		for i in range(len(self.choice)):
			if self.choice[i] not in self.word_id:
				self.word_id[self.choice[i]] = count
				count += 1
		print len(self.word_id)
		pickle.dump(self.word_id, open(self.path + "model/word_id", "wb"))
		fout = open(self.path + "/result/word_id", "wb")
		for key, value in self.word_id.iteritems() :
			fout.write("%s, %s\n" % (key, value))

	def select_corpus(self):
		self.select = []
		for i in range(len(self.x)):
			flag = 0
			for j in range(len(self.x[i])):
				if self.x[i][j] in self.y:
					flag = 1
					break
			if flag == 1:
				self.select.append(self.x[i])
			if len(self.select) >= 20000:
				break
		print "num = ", len(self.select)
		pickle.dump(self.select, open(self.path + "model/select_corpus", "wb"))
	
	def select_corpus_30(self):
		print "Start"
		self.step = 30
		self.data_size = 3591*self.each_num
		self.select = np.empty([self.data_size, self.step], dtype = int)
		self.y = np.empty([self.data_size], dtype = int) # id from 0 to 3591-1
		self.count = np.zeros(3591, dtype = int)
		self.pos = np.empty(self.data_size, dtype = int) # remove?
		temp = self.choice[:]
		count_sen = -1
		flag2 = 0
		for i in range(len(self.x)):
			flag = 0
			for j in range(len(self.x[i])-1, -1, -1):
				if j > 6 and self.x[i][j] in temp:
					flag = 1
					w = self.x[i][j]
					w_id = self.choice.index(w)
					self.count[w_id] += 1
					count_sen += 1

					self.pos[count_sen] = j
					if self.count[w_id] >= self.each_num:
						temp.remove(w)
						if len(temp) % 100 == 0: print "-", i, len(temp), w, w_id
						if len(temp) == 0: flag2 = 1
					break
					
			if flag == 1: # w is in s_i
				p = self.pos[count_sen]
				if p >= self.step:
					pad = p - self.step
					for j in range(self.step):
						self.select[count_sen][j] = self.dict[self.x[i][j+pad]] + 1
				else:
					pad = self.step - p
					for j in range(pad): self.select[count_sen][j] = 0
					for j in range(p):
						self.select[count_sen][j+pad] = self.dict[self.x[i][j]] + 1
						hot_id = self.choice.index(w)
				self.y[count_sen] = hot_id
			if flag2 == 1: break
		
		print "\n", count_sen
		print self.select[count_sen]
		print self.y[count_sen]
		print self.pos[count_sen]
		
		self.data_size = count_sen + 1
		self.select = self.select[:self.data_size]
		self.y = self.y[:self.data_size]
		pickle.dump(self.select, open(self.path + "model/x_train_260", "wb"))
		pickle.dump(self.y, open(self.path + "model/y_train_260", "wb"))
		self.output_count()
		print "data_size = ", self.data_size

	def output_count(self):
		fout = open(self.path + "/result/count260", "wb")
		print len(self.choice), len(self.count)
		for i in range(3591):
			fout.write("%s: %.0f\n" % (self.choice[i], self.count[i]))

	def produce_x(self):
		self.word_model = Word2Vec.load("../model/wordVector1")
		print "Construct x_train"
		N_maxWord = 30 # turn to 30~50
		print "data_size = ", self.data_size 
		self.x_train = np.zeros([self.data_size, N_maxWord, self.dim_size])

		for i in range(self.data_size): # embed first 1000 sentences
			for j in range(N_maxWord):
				if self.select[i][j] != ",,,":
					try:
						if j == 0: print self.select[i][j]
						self.x_train[i][j] = self.word_model[self.select[i][j]]
					except KeyError: 
						print "key", self.select[i][j]
		print "Pickle"
		pickle.dump(self.x_train[:self.output_data_size], open("../model/x_train_id", "wb"))
		
	def produce_y(self):
		self.vocab_dict = self.vocab_size
		self.y_train = np.zeros([self.data_size, 3591], dtype = float) # id from 0 to 3591-1

		print "Construct y_train"
		for i in range(self.data_size):
			print self.y[i]
			self.y_train[i][self.y[i]] = 1
		#pickle.dump(self.y_train[:self.output_data_size], open("../model/y_hot_vector2", "wb"))

	def shuffle_in_unison(self, a, b):
	    assert len(a) == len(b)
	    shuffled_a = np.empty(a.shape, dtype = a.dtype)
	    shuffled_b = np.empty(b.shape, dtype = b.dtype)
	    permutation = np.random.permutation(len(a))
	    for old_index, new_index in enumerate(permutation):
	        shuffled_a[new_index] = a[old_index]
	        shuffled_b[new_index] = b[old_index]
	    return shuffled_a, shuffled_b

	def build_y_test(self):
		print "Loading"
		self.choice_dict = pickle.load(open(self.path + "model/choice_list", "rb"))
		self.choice = pickle.load(open(self.path + "model/y_test_word", "rb"))
		self.y_test_id = np.empty([len(self.choice), 5], dtype = int)
		print self.choice.shape

		for i in range(len(self.choice)):
			for j in range(5):
				temp = self.choice[i][j]
				temp = temp.lower()
				temp = temp.replace("-", "")
				try:
					index = self.choice_dict.index(temp)
					self.y_test_id[i][j] = index
					print index
				except ValueError:
					self.y_test_id[i][j] = 0
					print i, j, self.choice[i][j]
		print self.y_test_id.shape
		print self.y_test_id[0:1]
		pickle.dump(self.y_test_id, open(self.path + "model/y_test_id2", "wb"))

def main():
	model = Model()
	model.load()
	#model.select_corpus()
	model.select_corpus_30()

	model.produce_x()
	model.produce_y()
	model.build_y_test()

if __name__ == '__main__':
	main()

