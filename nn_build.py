import tensorflow as tf
import numpy as np
import pickle
from segments import *

#load properties
with open('tmp/vars.pkl', 'rb') as f:
	batch_size, window_size, vocab_size, num_layers, from_indices = pickle.load(f)

#graph input
x = tf.placeholder(tf.int32, [batch_size, window_size])
y = tf.placeholder(tf.int32, [batch_size, window_size])

weights = {
	'out': tf.Variable(tf.random_normal([vocab_size, vocab_size]), name='weights')
}
biases = {
	'out': tf.Variable(tf.random_normal([vocab_size]), name='biases')
}

def RNN(x, weights, biases):
	lstm1 = tf.contrib.rnn.BasicLSTMCell(vocab_size)
	lstm2 = tf.contrib.rnn.BasicLSTMCell(vocab_size)
	lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2], state_is_tuple=True)
	initialState = lstm_cell.zero_state(batch_size, tf.float32)
	one_hot = tf.one_hot(x, vocab_size)
	output, nextState = tf.nn.dynamic_rnn(lstm_cell, one_hot, initial_state=initialState)
	rnn_logits = tf.contrib.layers.linear(output, vocab_size)
	logits_in = tf.reshape(rnn_logits, [-1, vocab_size])
	logits = tf.nn.xw_plus_b(logits_in, weights['out'], biases['out'])
	return tf.reshape(logits, [batch_size, window_size, vocab_size])
	
logits = RNN(x, weights, biases)

init = tf.global_variables_initializer()

#Building
saver = tf.train.Saver()
state = np.zeros([batch_size, window_size])
with tf.Session() as sess:
	saver.restore(sess, 'tmp/model.ckpt')
	i = 0
	while i < 200:
		x_batch = state
		res_probs = sess.run(logits, feed_dict = {x: x_batch})
		res_probs_batch = res_probs[0]
		results = np.argsort(-res_probs_batch[-1])
		j = 0
		while not is_valid(from_indices[state[0][-1]], from_indices[results[j]]):
			j += 1
		state[0] = np.append(state[0][1:], [results[j]])
		segment = segment_dict[from_indices[state[0][-1]]]
		print(segment)
		if segment == 'ELEM_BEGIN_STATION':
			break
		i += 1		
		
		
		
		
		