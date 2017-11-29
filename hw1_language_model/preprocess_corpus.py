# The first step
# Preprocess training data (corpus) from Holmos

from collections import Counter # remove infrequent words
from gensim.models import Word2Vec # word embedding
import numpy as np
import glob
import pickle

class Model(object):
	def __init__(self):
		self.number_file = 500 		# only process the number of files
		self.minimum_word = 4 		# the minimum length of the sentence
		self.th_infrequent = 11000 	# only process this number of frequent words
		self.dim_word2vec = 300 	# default = 300
		self.data_size = 20000
		self.vocab_size = 9779

		self.start_from_corpus = 0 # 1: start over, 0: read data from pickle
		self.corpus = []

		self.cut_path = "../data_cut"
		self.output_path = "../data"
		#self.stopwords = get_stop_words('english')
		#self.myStopwords = self.get_my_stop_words()

	def cut(self):
		fin = open("../Holmes_Training_Data/" + self.file_name, "r")
		flag = 0 # 0 to delete, 1 to save
		# save the processed file to the belowing path
		fout = open(self.cut_path + "/" + self.file_name, "wb")
			
		# start to cut
		for i, line in enumerate(fin):
			if flag == 0 and line[:21] == "*END*THE SMALL PRINT!": # cut the head
				flag = 1
				print "start:", i,

			elif flag == 1:
				if line[:20] == "End of Project Guten" or line[:20] == "End of The Project G": # cut the tail
					print "end:", i 
					break
				elif line[:7] == "Chapter" or line[:7] == "CHAPTER" or line[:4] == "PART":
					# cut chapter, part, content
					continue 
				elif line == "\r\n": # remove null sentence (can merge this func later)
					continue
				# elif line.count(" ") < 3: # remove sentence which is too short
				#	continue
				elif line[0] == " ":
					continue
				else: # the part to save
					fout.write("%s" %line)

	def remove_dot(self):
		fin = open(self.cut_path + "/" + self.file_name, "r")
		temp = ""
		
		for i, line in enumerate(fin):
			while True:
				flag = 0
				count1 = line.find(".")
				while True:
					if count1 != -1 and (line[count1-3:count1] == "Mrs" or line[count1-2:count1] == "Mr" \
						or line[count1-2:count1] == "Dr"):
						count1 = line.find(".", count1+1)
					else: break
				if count1 == -1:
					temp += line[:-2] + " "
					flag = 1 
						
				else:
					temp += line[:count1] # remove "."
					temp = temp.replace("\r\n", " ")
					
					while True: # remove blank in the beginning
						if len(temp) == 0 or temp[0] != " ": break # check here !!!
						temp = temp[1:]

					self.txt.append(temp)
					if line[count1+1:].find(".") == -1: # no more "."
						if line[count1+1] == '"': temp = line[count1+2:]
						else: temp = line[count1+1:]
						flag = 1
					else:
						line = line[count1+2:]
						temp = ""

				if flag == 1: break

	def remove_3punctuation(self):
		total = 0
		for index in range(len(self.txt)):
			temp = []
			w1 = 0
			i = index + total
			for j in range(len(self.txt[i])):
				w = self.txt[i][j]
				if w == '"' or w == '?' or w == '!':
					temp.append(self.txt[i][w1:j])
					w1 = j+1

			if len(temp) != 0:
				temp.append(self.txt[i][w1:])
				for k in range(temp.count('')):
					temp.remove('')
				if len(temp) > 0:
					self.txt[i] = temp[0]
					for k in range(len(temp)-1): self.txt.insert(i+k+1, temp[1+k])
					total += len(temp) - 1

	def check(self):
		while ' ' in self.txt: self.txt.remove(' ')
		
		for i in range(len(self.txt)):
			while True: # remove blank in the beginning
				if len(self.txt[i]) == 0 or self.txt[i][0] != " ": break # check here !!!
				self.txt[i] = self.txt[i][1:]
				if len(self.txt[i]) == 0: break

	def merge_sentence(self):
		self.txt = []
		self.remove_dot()
		self.remove_3punctuation() # '? ! "'
		self.check() # remove null sentence, the blank in the beginning

	def pre_sentence(self):
		for i in range(len(self.txt)):
			l = self.txt[i]
			l = l.lower()
			l = l.replace("-", " ")
			l = l.replace(",", " ")
			l = l.replace(":", " ")
			l = l.replace(";", " ")
			l = l.replace("[", " ")
			l = l.replace("]", " ")
			l = "".join([w for w in l if not w.isdigit()])
			l = [w for w in l.split()]
			#l = [w for w in l if w not in self.stopwords] 
			#l = [w for w in l if w not in self.myStopwords]
			self.txt[i] = l

	def remove_short(self, corpus):
		total = 0
		for index in range(len(corpus)):
			i = index - total
			if len(corpus[i]) <= self.minimum_word: # can change this number!!!
				corpus.remove(corpus[i])
				total += 1

	def output(self):
		fout = open(self.output_path + "/" + self.file_name, "wb")
		print len(self.txt)
		for i in range(len(self.txt)): fout.write("%s\n" %self.txt[i])

	def save_to_corpus(self):
		self.corpus += self.txt[10:-10]

	def pickle_corpus(self, save_pickle):
		print "Pickle corpus"
		if self.start_from_corpus == 0: save_pickle = 2

		if save_pickle == 1:
			pickle.dump(self.corpus, open("../model/corpus_500", "wb"))
		elif save_pickle == 2:
			self.corpus = pickle.load(open("../model/corpus_500", "rb"))

	def pickle_freq_corpus(self, save_pickle):
		print "Pickle frequent corpus"
		if save_pickle == 1: pickle.dump(self.corpus, open("../model/freq_corpus_500", "wb"))
		elif save_pickle == 2: self.corpus = pickle.load(open("../model/select_corpus", "rb"))
		for i in range(10): print self.corpus[i]
		print len(self.corpus)

	def add_common(self):
		fin = open("../result/add_to_common.txt", "r")
		for i, line in enumerate(fin):
			l = [w for w in line.split()]
			if l[1] not in self.common1:
				self.common1.append(l[1])
		for i in range(len(self.common1)):
			print self.common1[i]
		self.vocab_size = len(self.common1)
		print "vacob = ", self.vacob_size()

	def count_common(self):
		word_set = []
		for i in range(len(self.corpus)): word_set += self.corpus[i]
		# change [[1, 2, 3], [4, 5]] into [1, 2, 3, 4, 5]

		self.common1 = [] # save the most common word in order
		counts = Counter(word_set)
		commonTemp = counts.most_common(self.th_infrequent)
		for i in range(len(commonTemp)):
			self.common1.append(commonTemp[i][0])

		print "Common1.len = ", len(self.common1)
		self.add_common()
		pickle.dump(self.common1, open("../model/common1_word", "wb"))

	def reset_corpus(self):
		print "Reset corpus (frequent)"
		for i in range(len(self.corpus)):
			self.corpus[i] = [w for w in self.corpus[i] if w in self.common1]
		self.remove_short(self.corpus)

		fout = open(self.output_path + "/test1", "wb")
		for i in range(200):
			fout.write("%s\n" %self.corpus[i]) 

	def remove_infrequent(self):
		print "Remove infrequent words"
		self.count_common()
		#self.common1 = pickle.load(open("../model/common1_word", "rb"))
		self.reset_corpus()

	def construct_embedding(self, run_again):
		print "Construct embedding mapping"
		if run_again == 1:
			self.word_model = Word2Vec(self.corpus, size = self.dim_word2vec, window = 15, min_count = 1, workers = 2, iter = 15)
			self.word_model.init_sims(replace = True)
			self.word_model.save("../model/wordVector1")
		else:
			self.word_model = Word2Vec.load("../model/wordVector1")  # you can continue training with the loaded model!
		
		print self.word_model.similarity('he', 'she')
		print self.word_model.most_similar_cosmul(positive = ['his', 'she'], negative = ['he'], topn = 2)
		print self.word_model.most_similar_cosmul(positive = ['man', 'she'], negative = ['he'], topn = 2)
		print self.word_model.most_similar_cosmul(positive = ['boy', 'she'], negative = ['he'], topn = 2)
		print self.word_model.most_similar_cosmul(positive = ['mr.', 'she'], negative = ['he'], topn = 2)
		#print wordModel.similarity('mac', 'os')
		#print self.wordModel.similar_by_word('mac', topn = 10, restrict_vocab = None)
		
	def word2vec(self):
		print "Word Embedding"
		N_maxWord = 30 # turn to 30~50
		self.word_vector = np.zeros([self.data_size, N_maxWord, self.dim_word2vec])
		#self.noVector = []

		for index in range(self.data_size): # embed first 1000 sentences
			i = 1 * index
			temp = np.zeros([N_maxWord, self.dim_word2vec])
			pad_size = N_maxWord - len(self.corpus[i])
			if pad_size <= 0: 
				pad_size = 0
				input_len = N_maxWord
			else: input_len = len(self.corpus[i])
			for j in range(input_len):
				try:
					temp[j + pad_size] = self.word_model[self.corpus[i][j]]
				except KeyError: # may change later
					continue
	
			self.word_vector[index] = np.array(temp)
		pickle.dump(self.word_vector, open("../model/wordEmbed1", "wb"))
		#else:
		#	self.titleVector = pickle.load(open("model-titleVector6", "rb"))

	def construct_vocab_dict(self):
		self.common1 = pickle.load(open("../model/common1_word", "rb"))
		print "Construct dictionary"
		self.vocab_dict = {}
		for i in range(self.vocab_size):
			self.vocab_dict[self.common1[i]] = i
		print len(self.vocab_dict)
		pickle.dump(self.vocab_dict, open("../model/dict", "wb"))

	def produce_y(self):
		self.vocab_dict = self.vocab_size
		self.num_steps = 30
		
		print "Loading Dictionary"
		self.corpus = pickle.load(open("../model/select_corpus", "rb"))
		print len(self.corpus)
		self.dict = pickle.load(open("../model/dict", "rb"))

		print "Construct y_train"
		self.y_train = np.zeros((self.data_size, self.vocab_size), dtype = float)
		for index in range(self.data_size):
			i = index * 1
			if len(self.corpus[i]) >= self.num_steps + 1:
				index = self.dict[self.corpus[i][self.num_steps]] # c[20] is the output
			else:
				#padding = self.num_steps - len(self.corpus[i])
				try:
					n = self.corpus[i][-1]
				except IndexError:
					print i, len(self.corpus[i])
					print self.corpus[i]
				ind = self.dict[n] # the last word is the output
			self.y_train[index][ind] = 1
			if i < 10: print ind
		pickle.dump(self.y_train, open("../model/y_hot_vector", "wb"))

def main():
	model = Model()
	#model.common1 = pickle.load(open("../model/common1_word", "rb"))
	model.start_from_corpus = 1
	if model.start_from_corpus == 1:
		for i, name in enumerate(glob.glob('../Holmes_Training_Data/*.TXT')):
		    if i >= model.number_file: break
		    
		    # get the file name
		    model.file_name = name[24:]
		    model.index = i
		    print "Book:", i, model.file_name
		    
		    model.cut() # cut head and tail
		    model.merge_sentence()
		    model.pre_sentence()
		    model.remove_short(model.txt) # remove the sentences which are too short 
		    model.save_to_corpus() # merge all txt to corpus
		    print len(model.corpus)
		#model.pickle_corpus(1)
	mode = 1
	if mode == 1:    
		#model.pickle_corpus(2) # save to pickle, 1: save, 2: read, 0: nothing
		model.remove_infrequent()
		model.pickle_freq_corpus(1) # 1: save, 2: read, 0: nothing
		model.construct_embedding(2) # 1: run again, other: nothing
		model.word2vec()
		model.construct_vocab_dict()
		model.produce_y()
		
	# construct x_train 3D-array 1000*20*300, y_train 1000*8000
	if mode == 0:
		model.pickle_freq_corpus(2) # 1: save, 2: read, 0: nothing
		print "corpus = ", len(model.corpus)
		model.construct_embedding(2) # 1: run again, other: nothing
		#model.word2vec()
		
	# construct the dic only
	#model.pickle_corpus(2)
	#model.count_common()

if __name__ == '__main__':
	main()

    


    
    
    
