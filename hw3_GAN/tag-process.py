from numpy import genfromtxt
import numpy as np
import pandas as pd
import pickle

def load():
	raw_data = pd.read_csv('../data/tags_clean.csv', header = None).as_matrix()
	print "raw_data.shape = ", raw_data.shape
	#print raw_data[0, :]

	data = raw_data[:, 1]
	print "data.shape = ", data.shape # = 18*5760
	return data

def count():
	count_hair, count_eyes, count_total = 0, 0, 0

	for i in range(data.shape[0]):
		#if "hair" in data[i]: count_hair += 1
		#if "eyes" in data[i]: count_eyes += 1
		if "hair" in data[i] and "eyes" in data[i]:
			count_total += 1
			data_h_e.append([i, data[i]])
			
	#print "count = ", count_hair, count_eyes, 
	print "count_total =", count_total

def split_number(mylist):
	for i in range(len(mylist)): mylist[i] = [w for w in mylist[i].split(":")]
	max_num, max_idx = 0, -1
	
	for i in range(len(mylist)):
		if int(mylist[i][1]) > int(max_num):
			max_idx = i
			max_num = mylist[i][1]
			
	try: t = mylist[max_idx][0].split(" ")
	except IndexError: print "IndexError", max_idx
	
	return t[0]
	
def hair_color():
	tags = []
	i = 0
	for raw_tag in data_h_e:
		idx, tag = raw_tag
		
		temp = []
		tag = [w for w in tag.split("\t")]
		
		for j in range(len(tag)):
			if "hair" in tag[j] and "long" not in tag[j] and "short" not in tag[j] and "pubic" not in tag[j]: 
				temp.append(tag[j])
		if len(temp) == 0: continue
		tags.append([idx])
		tags[i].append(split_number(temp))

		temp = []
		for j in range(len(tag)):
			if "eyes" in tag[j] and "11" not in tag[j]:
				temp.append(tag[j])
		if len(temp) == 0: 
			tags.remove(tags[-1])
			continue
				
		tags[i].append(split_number(temp))
 		
		i += 1
		
	print "first tags -- "
	for i in range(10):
		print tags[i]
	
	print "data.len =", len(tags)
	return tags

def tag_to_id():
	hair_list = []
	eyes_list = []
	hair_dict = {}
	eyes_dict = {}
	np.save("../data/my_tags_word", data)
	
	for i in range(len(data)):
		if data[i][1] not in hair_list: 
			hair_dict[data[i][1]] = len(hair_list)
			hair_list.append(data[i][1])
		if data[i][2] not in eyes_list: 
			eyes_dict[data[i][2]] = len(eyes_list)
			eyes_list.append(data[i][2])
	print "hair/eyes.len =", len(hair_dict), len(eyes_dict)
	print "hair =", hair_dict
	print "eyes =", eyes_dict

	for i in range(len(data)):
		data[i][1] = hair_dict[data[i][1]]
		data[i][2] = eyes_dict[data[i][2]]
		
	pickle.dump(hair_dict, open("../data/hair_dict", "wb")) 
	pickle.dump(eyes_dict, open("../data/eyes_dict", "wb")) 
	return data

def save():
	data_arr = np.array(data)
	print data_arr.shape
	for i in range(30): print data[i]
	np.save("../data/my_tags", data_arr)

def main():
	data = load()
	data_h_e = []
	count()
	data = hair_color()
	data = tag_to_id()
	save()

if __name__ == '__main__':
	main()
