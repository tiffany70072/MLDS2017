import numpy as np
import json
import sys
import os

TEXT_DATA_DIR = "MLDS_hw2_data/training_data/feat/"

all_feat = []
all_label = dict()

with open("MLDS_hw2_data/training_label.json") as json_data:
	label = json.load(json_data)
	
for i in range(len(label)):
	all_label[label[i]["id"]] = label[i]["caption"]

for feat in os.listdir(TEXT_DATA_DIR):
	idx = feat.strip().split(".npy")[0]
	print idx
	print all_label[idx][0]
	exit()
	path = os.path.join(TEXT_DATA_DIR, feat)
	try:
		data = np.load(path)
		all_feat.append(data)
	except:
		continue

print np.array(all_feat).shape


