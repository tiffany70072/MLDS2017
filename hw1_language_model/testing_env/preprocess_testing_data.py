from stop_words import get_stop_words
from collections import Counter 	# remove infrequent words
from gensim.models import Word2Vec 	# word embedding
from numpy import genfromtxt
import numpy as np
import pickle
import csv

class Model(object):
	def __init__(self):
		self.dim_word2vec = 300 # default = 300
		self.N_max = 30
		self.back = 2

		self.path = "../"
		self.file_name = '../testing_data.csv'
		self.stopwords = get_stop_words('english')
		self.myStopwords = self.get_my_stop_words()

	def get_my_stop_words(self):
		f_stopwords = open(self.path + "stop_words.txt", "r") 
		self.myStopwords = []  
		for line in f_stopwords: self.myStopwords.append(line[:-1])	
		f_stopwords.close()
		return self.myStopwords

	def cut(self):
		fin = self.file_name
		with open(fin, "r") as f:
			raw_data = csv.reader(f)
			raw_data = list(raw_data)
			raw_data = raw_data[1:len(raw_data)]
		raw_data = np.array(raw_data)

		print "raw_data.shape = ", raw_data.shape
		print raw_data[1]

		self.corpus = raw_data[:, 1]
		self.choice = raw_data[:, 2:]

		pickle.dump(self.corpus, open(self.path + "model/x_test_word", "wb"))
		pickle.dump(self.choice, open(self.path + "model/y_test_word", "wb"))
	
	def load_xy(self):
		self.corpus = pickle.load(open(self.path + "model/x_test_word", "rb"))
		self.choice = pickle.load(open(self.path + "model/y_test_word", "rb"))

	def load(self):
		self.dict = pickle.load(open(self.path + "model/dict", "rb"))
		self.common1 = pickle.load(open(self.path + "model/common1_word", "rb"))

	def pre_sentence(self):
		print "Pre_sentence"

		temp = []
		for i in range(len(self.corpus)):
			l = self.corpus[i]
			l = l.lower()
			l = l.replace("-", " ")
			l = l.replace(".", " ")
			l = l.replace(",", " ")
			l = l.replace(":", " ")
			l = l.replace(";", " ")
			l = l.replace("[", " ")
			l = l.replace("]", " ")
			l = [w for w in l.split()]
			#l = [w for w in l if w not in self.stopwords] 
			#l = [w for w in l if w not in self.myStopwords]
			temp.append(l)
		self.corpus = temp
		print "Corpus len = ", len(self.corpus)

	def remove_short(self, corpus):
		total = 0
		for index in range(len(corpus)):
			i = index - total
			if len(corpus[i]) <= self.minimum_word: # can change this number!!!
				corpus.remove(corpus[i])
				total += 1
		
	def remove_infrequent(self):
		print "Remove infrequent words"
		self.common1.append("_____")
		print "Reset corpus (frequent)"
		for i in range(len(self.corpus)):
			self.corpus[i] = [w for w in self.corpus[i] if w in self.common1]
		
		fout = open(self.path + "result/temp2", "wb")
		for i in range(200):
			fout.write("%s\n" %self.corpus[i]) 

	def get_pos(self):
		self.pos = np.empty((len(self.corpus)), dtype = int)
		for i in range(len(self.corpus)):
			try:
				temp = self.corpus[i].index("_____")
				self.pos[i] = temp
				if i<10: print temp
	
			except ValueError: 
				print "Error!", i
		
	def word2vec(self):
		print "Loading word2vec model"
		self.word_model = Word2Vec.load(self.path + "model/wordVector1")
		
		print "Word Embedding"
		N_maxWord = self.N_max # turn to 30~50
		N_back = 2
		self.x = np.zeros([len(self.corpus), N_maxWord+N_back, self.dim_word2vec])

		for i in range(len(self.corpus)): # embed first 1000 sentences
			temp = np.zeros([N_maxWord+N_back, self.dim_word2vec])
			
			front = self.pos[i]
			if front <= N_maxWord:
				pad = N_maxWord - front # start from here
				for j in range(front): temp[j+pad] = self.word_model[self.corpus[i][j]] # 0 ~ p-1
			else:
				pad = front - N_maxWord
				for j in range(N_maxWord): temp[j] = self.word_model[self.corpus[i][j+pad]]

			back = len(self.corpus[i]) - self.pos[i] - 1 
			if back < N_back:
				for j in range(back):
					temp[N_maxWord+j] = self.word_model[self.corpus[i][j+front+1]]
			else:
				for j in range(N_back):
					temp[N_maxWord+j] = self.word_model[self.corpus[i][j+front+1]]
			
			self.x[i] = temp
		pickle.dump(self.x, open(self.path + "model/x_test", "wb"))
		print self.x.shape
				
	def y_test(self):
		self.word_model = Word2Vec.load(self.path + "model/wordVector1")
		self.y_test = np.zeros([len(self.choice), 5, self.dim_word2vec])
		total = 0
		for i in range(len(self.choice)):
			for j in range(5):
				temp = self.choice[i][j]
				temp = temp.lower()
				temp = temp.replace("-", "")
				if temp not in self.common1:
					print i, j, temp
					temp = "the"
					total += 1
				try:
					self.y_test[i][j] = self.word_model[temp]
				except KeyError:
					print i, j, temp
		print "t = ", total
		print self.y_test.shape
		print self.y_test[0, :, :5]
		pickle.dump(self.y_test, open(self.path + "model/y_test", "wb"))

	def output(self, txt, name):
		fout = open(self.path + "/result/temp" + str(name), "wb")
		for i in range(len(txt)):
			fout.write("%s\n" % txt[i])

	def y_id(self):
		print "\ny_id"
		self.y_test_id = np.empty([len(self.choice), 5], dtype = int)
		for i in range(len(self.choice)):
			for j in range(5):
				try:
					index = self.dict[self.choice[i][j]]
					self.y_test_id[i][j] = index
				except KeyError:
					self.y_test_id[i][j] = 0
					print i, j, self.choice[i][j]
		print self.y_test_id.shape
		print self.y_test_id[0:1]
		pickle.dump(self.y_test_id, open(self.path + "model/y_test_id", "wb"))

	def x_test(self):
		run_again = 1
		if run_again == 1:
			self.pre_sentence()
			#self.output(self.corpus, 3)
			#self.remove_infrequent()
			#self.output(self.corpus, 4)
			#self.get_pos()
			#self.output(self.pos, 5)
			pickle.dump(self.corpus, open(self.path + "model/x_test_word3", "wb"))
			#pickle.dump(self.pos, open(self.path + "model/pos", "wb"))
		else:
			self.corpus = pickle.load(open(self.path + "model/x_test_word2", "rb"))
			self.pos = pickle.load(open(self.path + "model/pos", "rb"))

		#self.word2vec()

	def x_id(self):
		print "\nx_id"
		self.corpus = pickle.load(open(self.path + "model/x_test_word2", "rb"))
		self.pos = pickle.load(open(self.path + "model/pos", "rb"))
		self.dict = pickle.load(open(self.path + "model_wen/my_wordid", "rb"))
		self.x_test_id = np.zeros([1040, self.back], dtype = int)
		print len(self.pos)

		for i in range(1040):
			if len(self.corpus[i]) - self.pos[i] <= 2: 
				for k in range(self.back): self.corpus[i].append("the")

			for j in range(self.back):
				try:
					index = self.dict[self.corpus[i][self.pos[i]+j+1]] + 1 # for wordid
					self.x_test_id[i][j] = index
				except KeyError:
					self.x_test_id[i][j] = 0
					print i, j, self.corpus[i][self.pos[i]+j+1] + 1

		print self.x_test_id.shape
		print self.x_test_id[:self.N_max]
		pickle.dump(self.x_test_id, open(self.path + "model/x_test_id", "wb"))

	def x_test_hot(self):
		self.N_maxWord = 30
		self.corpus = pickle.load(open(self.path + "model/x_test_word2", "rb"))
		self.pos = pickle.load(open(self.path + "model/pos", "rb"))
		self.dict = pickle.load(open(self.path + "model_wen/my_wordid", "rb"))
		self.x_test = np.zeros([1040, self.N_maxWord], dtype = int)
		for i in range(1040):
			front = self.pos[i]
			if front <= self.N_maxWord:
				pad = self.N_maxWord - front # start from here
				for j in range(front): 
					try:
						self.x_test[i][j+pad] = self.dict[self.corpus[i][j]] + 1
					except KeyError:
						print self.corpus[i][j]
			else:
				pad = front - self.N_maxWord
				for j in range(self.N_maxWord): 
					try:
						self.x_test[i][j] = self.dict[self.corpus[i][j+pad]] + 1
					except KeyError:
						print self.corpus[i][j]
			if i == 100:
				print front
				print pad
				print self.corpus[i]
				print self.x_test[i]
		pickle.dump(self.x_test, open(self.path + "model_/x_test_hot", "wb"))
		print self.x_test.shape

	def y_test_hot(self):
		self.dict = pickle.load(open(self.path + "model/choice_list", "rb"))
		self.choice = pickle.load(open(self.path + "model/y_test_word", "rb"))
		self.y_test = np.zeros([len(self.choice), 5])
		total = 0
		for i in range(len(self.choice)):
			for j in range(5):
				temp = self.choice[i][j]
				temp = temp.lower()
				temp = temp.replace("-", "")
				
				try:
					index = self.dict.index(temp)
					self.y_test[i][j] = index
				except ValueError:
					print i, j, self.choice[i][j], temp
		
		print self.y_test.shape
		print self.y_test[100]
		pickle.dump(self.y_test, open(self.path + "model/y_test_hot", "wb"))

def main():
	model = Model()
	run_again = 0
	if run_again == 1: model.cut()
	else: model.load_xy()
	#model.load()

	model.x_test()
	model.y_test()
	#model.y_id()
	#model.x_id()

	model.y_test_hot()

if __name__ == '__main__':
	main()
