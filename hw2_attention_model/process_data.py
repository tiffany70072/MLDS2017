# Preprocess training data and public testing data

import numpy as np
import pickle
import json
import sys
import os
from tempfile import TemporaryFile

TEXT_DATA_DIR = "../MLDS_hw2_data/training_data/feat/"
TEXT_DATA_DIR_TEST = "../MLDS_hw2_data/testing_data/feat/"

all_label = []
id_to_index = []

#with open("../MLDS_hw2_data/training_label.json") as json_data:
with open("../MLDS_hw2_data/testing_public_label.json") as json_data:
	label = json.load(json_data)
	
'''for i in range(len(label)):
	all_label.append(label[i]["caption"])
	id_to_index.append(label[i]["id"])
'''

#pickle.dump(all_label, open("../model/all_label", "wb"))

all_feat = np.empty([len(label), 80, 4096], dtype = float)

#for feat in os.listdir(TEXT_DATA_DIR):
for feat in os.listdir(TEXT_DATA_DIR_TEST):
	idx = feat.strip().split(".npy")[0]
	path = os.path.join(TEXT_DATA_DIR, feat)
	
	try:
		idx = id_to_index.index(idx)
		data = np.load(path)
		all_feat[idx] = data

	except:
		continue

#outfile = TemporaryFile()
#np.save("../model/all_feat", all_feat)
print all_feat
np.save("../testing_src/feat_test", all_feat)

