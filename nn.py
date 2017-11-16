import tensorflow as tf
import sys
import os
import numpy as np
import random
from segments import *

batch_size = 5
window_size = 50
num_epochs = 100
num_layers = 2

def preprocess(dir, files):
	data = []
	for file in files:
		with open('{0}/{1}'.format(dir, file), 'rb') as f:
			data += f.read()
	data_cleaned = clean(data)
	data_padded = [0xff] * window_size + data_cleaned
	return data_padded

def indexed(data):
	indices = dict()
	indices[0xff] = 0
	i = 1
	for segment in data:
		if segment not in indices:
			indices[segment] = i
			i += 1
	inverse_indices = dict(zip(indices.values(), indices.keys()))
	indexed = [indices[segment] for segment in data if segment in indices]
	vocab_size = len(indices)
	return indexed, indices, inverse_indices, vocab_size
	
def separate_training(data, train_samples):
	j = -1
	for i in range(train_samples):
		while data[j-1] != 0:
			j -= 1
		while data[j-1] == 0:
			j -= 1
	return data[:j], data[j:]
	
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
	

def next_piece(sess, state):
	inpts = state
	res_probs = sess.run(logits, feed_dict = {inpt: inpts})
	res_probs_batch = res_probs[0]
	results = np.argsort(-res_probs_batch[-1])
	j = 0
	while not is_valid(inverse_indices[state[0][-1]], inverse_indices[results[j]]):
		j += 1
	state[0] = np.append(state[0][1:], [results[j]])
	segment = segment_dict[inverse_indices[state[0][-1]]]
	return segment, state

def train_model(data, batch_size, window_size, num_epochs):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		print('Begin training')
		x, y = batched(data, batch_size, window_size)	
		for epoch in range(num_epochs):
			avg_loss = 0
			loss_count = 0
			for i in range(len(x)):
				inpts = x[i]
				answrs = y[i]
				_, loss_val = sess.run([train, loss], feed_dict = {inpt: inpts, answr: answrs})
				avg_loss += loss_val
				loss_count += 1
			avg_loss /= float(loss_count)
			print('{0}: {1}'.format(epoch, avg_loss))
				
		save_path = saver.save(sess, 'tmp/model.ckpt')
		print('Model saved to file {0}'.format(save_path))
	
def build(batch_size, window_size):
	with tf.Session() as sess:
		saver.restore(sess, 'tmp/model.ckpt')
			
		print('Begin building')
		state = np.zeros([batch_size, window_size])
		i = 0
		while i < 100:
			segment, state = next_piece(sess, state)
			print(segment)
			if segment == 'ELEM_BEGIN_STATION':
				break
			i += 1
			
if len(sys.argv) != 3:
	sys.exit('Invalid syntax. Use nn [dirname] [train, test]')
dir = sys.argv[1]

files = os.listdir(dir)
random.shuffle(files)
data = preprocess(dir, files)
data_indexed, indices, inverse_indices, vocab_size = indexed(data)			
			
#Begin model
inpt = tf.placeholder(tf.int32, shape = [batch_size, window_size])
answr = tf.placeholder(tf.int32, shape = [batch_size, window_size])
hidden = tf.one_hot(inpt, vocab_size)

def lstm_cell():
	return tf.contrib.rnn.BasicLSTMCell(vocab_size)
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
initialState = cell.zero_state(batch_size, tf.float32)
output, nextState = tf.nn.dynamic_rnn(cell, hidden, initial_state = initialState)

W = tf.get_variable('W', [vocab_size, vocab_size])
b = tf.get_variable('b', [vocab_size])

output = tf.reshape(output, [-1, vocab_size])
logits_1 = tf.nn.xw_plus_b(output, W, b)
logits = tf.reshape(logits_1, [batch_size, window_size, vocab_size])

xEnt = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = answr)
loss = tf.reduce_sum(xEnt)
train = tf.train.AdamOptimizer(.01).minimize(loss)

saver = tf.train.Saver()			
			
if sys.argv[2] == 'train':
	train_model(data_indexed, batch_size, window_size, num_epochs)
	build(batch_size, window_size)
elif sys.argv[2] == 'test':
	build(batch_size, window_size)
else:
	sys.exit('Invalid syntax. Use nn [dirname] [train, test]')