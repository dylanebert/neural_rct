#To train the RNN. Train results saved in tmp (overwrites)

import tensorflow as tf
import numpy as np
import sys
import pickle
from preprocess import preprocess, batched

#hyperparameters
batch_size = 25
window_size = 50
num_epochs = 100
num_layers = 2

#data
if len(sys.argv) != 2:
	sys.exit('Invalid syntax. Use \'nn [dirname]\' for training')
data, from_indices, vocab_size = preprocess(sys.argv[1], window_size)
x_batches, y_batches = batched(data, batch_size, window_size)
num_batches = len(x_batches)

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
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(.01).minimize(loss)

init = tf.global_variables_initializer()

#Training
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init)
	
	for step in range(num_epochs):
		perplexity = 0
		for i in range(num_batches):
			x_batch = x_batches[i]
			y_batch = y_batches[i]
			_, loss_val = sess.run([optimizer, loss], feed_dict={x: x_batch, y: y_batch})
			perplexity += loss_val
		perplexity = np.exp(perplexity / num_batches)
		print(perplexity)
		
	saver.save(sess, 'tmp/model.ckpt')
	
#Save properties
with open('tmp/vars.pkl', 'wb') as f:
	pickle.dump([batch_size, window_size, vocab_size, num_layers, from_indices], f)