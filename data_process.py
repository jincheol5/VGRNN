import numpy as np
import torch
import pickle


# Enron dataset
with open('data/enron10/adj_time_list.pickle', 'rb') as handle:
    adj_time_list = pickle.load(handle,encoding='latin1')

with open('data/enron10/adj_orig_dense_list.pickle', 'rb') as handle:
    adj_orig_dense_list = pickle.load(handle,encoding='latin1')


print(type(adj_orig_dense_list))