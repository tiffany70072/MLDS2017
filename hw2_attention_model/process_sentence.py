from tempfile import TemporaryFile
import numpy as np
import pickle
import json
import sys
import os

sen = pickle.load(open("../model/all_label", "rb"))
sen_id = pickle.load(open("../model/all_label_preprocess_idx", "rb"))
new = []
total_id = 0
total_sen = 0
total = 0
id_list = []

for i in range(len(sen)):
	total += len(sen[i])
	c = 0
	l = []
	l_remove = []
	for j in range(len(sen[i])):
		s = sen[i][j].split()

		if len(s) < 6:
			l_remove.append(j)

		else:
			try:
				if ((s[0].lower() == "a" or s[0].lower == "an") and len(s) > 2 and (s[2] == "is" or s[3] == "is"))\
					or (s[0].lower() == "the" and len(s) > 2 and (s[2] == "is" or s[3] == "is"))\
					or ((s[0][-1] == "s" or s[0][-3:] == "men") and len(s) > 1 and s[1] == "are")\
					or ((s[1][-1] == "s" or s[1][-3:] == "men") and len(s) > 2 and s[2] == "are")\
					or ("and" in s)\
					or ("on" in s)\
					or ("into" in s)\
					or ("in" in s)\
					or ("at" in s)\
					or ("with" in s)\
					or ("before" in s)\
					or ("after" in s)\
					or ("inside" in s)\
					or ("out" in s)\
					or ("under" in s)\
					or (len(s)>11):
					c += 1
					l.append(j)

			except IndexError:
				print i, j, sen[i][j], len(s)
			
	if c > 2: 
		total_id += 1
		temp = []
		for idx in l: 
			temp.append(sen[i][idx])
			total_sen += 1
			
		new.append(temp)
		id_list.append(l)
		
	else: 
		temp = []
		l = []
		for idx in range(len(sen[i])):
			if idx not in l_remove: 
				temp.append(sen[i][idx])
				l.append(idx)
				total_sen += 1
		id_list.append(l)
		new.append(temp)

print "total_id =", total_id, ", sen =", total_sen
print "%.5f " %(total_sen/float(total))

pickle.dump(id_list, open("../model/sen_list", "wb"))

new = []
for i in range(len(sen_id)):
	temp = []
	for idx in range(len(id_list[i])): temp.append(sen_id[i][idx])
	new.append(temp)

pickle.dump(new, open("../model/all_label_select_id2", "wb"))
