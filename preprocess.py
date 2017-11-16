import os
import sys
import numpy as np
import random
from segments import clean

#Takes directory and window size, returns indexed data, dictionary to revert back from indices, and vocab size
def preprocess(dir, window_size):
	data = []
	files = os.listdir(dir)
	for file in files:
		with open('{0}/{1}'.format(dir, file), 'rb') as f:
			data += [0xff] * window_size
			data += f.read()
	data = clean(data) #Map covered to normal
	
	to_indices = dict() #Map to indices
	to_indices[0xff] = 0
	i = 1
	for segment in data:
		if segment not in to_indices:
			to_indices[segment] = i
			i += 1
	from_indices = dict(zip(to_indices.values(), to_indices.keys())) #Map from indices to segment values
	indexed = [to_indices[segment] for segment in data if segment in to_indices]
	vocab_size = len(to_indices)
	return indexed, from_indices, vocab_size
	
def batched(data, batch_size, window_size):
	num_batches = len(data) // batch_size
	data = np.reshape(data[0 : batch_size * num_batches], [batch_size, num_batches])
	epoch_size = (num_batches - 1) // window_size
	x = list()
	y = list()
	for i in range(epoch_size):
		x.append(data[:, i * window_size : (i + 1) * window_size])
		y.append(data[:, i * window_size + 1 : (i + 1) * window_size + 1])
	x = np.array(x, dtype=np.uint8)
	y = np.array(y, dtype=np.uint8)
	return x, y