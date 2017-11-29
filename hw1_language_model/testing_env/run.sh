#!/bin/bash

python2 preprocess_testing_data.py $1
python2 lstm_one_hot_testing.py $2