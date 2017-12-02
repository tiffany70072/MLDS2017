import numpy as np

corpus = np.load("../data/" + "corpus_15.npy")
print "corpus =", corpus.shape

num_steps = 15
count = 0

for i in range(corpus.shape[0]):
	try:
		n = np.where(corpus[i, 1] == 3)[0][0]
		for j in range(n, num_steps-1): corpus[i][1][j] = corpus[i][1][j+1]
		corpus[i][1][j+1] = 0
		count += 1
	except IndexError: continue
	
print count
np.save("../data/corpus_15_remove", corpus)
