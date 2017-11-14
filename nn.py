import tensorflow as tf
import sys
import os
import numpy as np
from segments import segment_dict

batch_size = 20
window_size = 20
num_epochs = 50
num_layers = 2

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

if len(sys.argv) != 2:
	sys.exit('Invalid syntax. Use nn [dirname]')
dir = sys.argv[1]
files = os.listdir(dir)
data = []
for file in files:
	with open('{0}/{1}'.format(dir, file), 'rb') as f:
		data += f.read()
data_indexed, indices, inverse_indices, vocab_size = indexed(data)

inpt = tf.placeholder(tf.int32, shape = [batch_size, window_size])
answr = tf.placeholder(tf.int32, shape = [batch_size, window_size])
hidden = tf.one_hot(inpt, vocab_size)

def lstm_cell():
	return tf.contrib.rnn.BasicLSTMCell(vocab_size)
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
initialState = cell.zero_state(batch_size, tf.float32)
output, nextState = tf.nn.dynamic_rnn(cell, hidden, initial_state = initialState)

W = tf.Variable(tf.random_normal([vocab_size, vocab_size], stddev = .1))
b = tf.Variable(tf.random_normal([vocab_size], stddev = .1))

output = tf.reshape(output, [-1, vocab_size])
logits_1 = tf.nn.xw_plus_b(output, W, b)
logits = tf.reshape(logits_1, [batch_size, window_size, vocab_size])

xEnt = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = answr)
loss = tf.reduce_sum(xEnt)
train = tf.train.AdamOptimizer(.01).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
	
x, y = batched(data_indexed, batch_size, window_size)	
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
		
'''print('Begin testing')
avg_loss = 0
loss_count = 0
x, y = batched(tracks_test, batch_size, window_size)
for i in range(len(x)):
	inpts = x[i]
	answrs = y[i]
	loss_val = sess.run(loss, feed_dict = {inpt: inpts, answr: answrs})
	avg_loss += loss_val
	loss_count += 1
avg_loss /= float(loss_count)
print(avg_loss)'''

print('Begin building')
#state = np.zeros([batch_size, window_size])
state = x[0]
i = 0
while i < 100:
	inpts = state
	res = sess.run(logits, feed_dict = {inpt: inpts})
	state = [[np.argmax(res[j][k]) for k in range(len(state[j]))] for j in range(len(state))]
	try:
		print(segment_dict[inverse_indices[state[0][-1]]])
	except:
		print('{0:x}'.format(inverse_indices[state[0][-1]]))
	i += 1