from tempfile import TemporaryFile
import numpy as np
import pickle
import json
import sys
import os

word_id = pickle.load(open("../model/word_to_index", "rb"))
id_word = {}

fout = open('../temp/word_id','w')
for key, value in word_id.iteritems():
	id_word[value] = key
	fout.write(key + " " + str(value) + "\n")
fout.close()

fout = open('../temp/id_word','w')
for key, value in id_word.iteritems():
	fout.write(str(key) + " " + value + "\n")
fout.close()

pickle.dump(id_word, open("../model/id_word", "wb"))
    