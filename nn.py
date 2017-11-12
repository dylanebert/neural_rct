from read_td6 import *
import tensorflow as tf
import sys
import numpy as np

batch_size = 5
window_size = 25
embed_size = 100
num_epochs = 100

def indexed(data):
	indices = dict()
	i = 0
	for segment in data:
		if segment not in indices:
			indices[segment] = i
			i += 1
	indexed = [indices[segment] for segment in data if segment in indices]
	return indexed, indices, len(indices)
	
def batched(data, batch_size, window_size):
	num_batches = len(data) // batch_size	
	data = np.reshape(data[0 : batch_size * num_batches], [batch_size, num_batches])	
	epoch_size = (num_batches - 1) // window_size	
	x = list()
	y = list()
	for i in range(epoch_size):
		x.append(data[:, i * window_size : (i + 1) * window_size])
		y.append(data[:, i * window_size + 1 : (i + 1) * window_size + 1])
	x = np.array(x)	
	y = np.array(y)
	return x, y

dir = sys.argv[1]
tracks_raw = get_raw_data(dir)
tracks, indices, vocab_size = indexed(tracks_raw)
inverse_indices = dict(zip(indices.values(), indices.keys()))

tracks_train = tracks[:-100]
tracks_test = tracks[-100:]

inpt = tf.placeholder(tf.int32, shape = [batch_size, window_size])
answr = tf.placeholder(tf.int32, shape = [batch_size, window_size])
E = tf.Variable(tf.random_normal([vocab_size, embed_size], stddev = .1))
embeddings = tf.nn.embedding_lookup(E, inpt)

rnn = tf.contrib.rnn.BasicLSTMCell(embed_size)
initialState = rnn.zero_state(batch_size, tf.float32)
output, nextState = tf.nn.dynamic_rnn(rnn, embeddings, initial_state = initialState)

W = tf.Variable(tf.random_normal([embed_size, vocab_size], stddev = .1))
b = tf.Variable(tf.random_normal([vocab_size], stddev = .1))

output = tf.reshape(output, [-1, embed_size])
logits_1 = tf.nn.xw_plus_b(output, W, b)
logits = tf.reshape(logits_1, [batch_size, window_size, vocab_size])

xEnt = tf.contrib.seq2seq.sequence_loss(logits, answr, tf.ones([batch_size, window_size], dtype = tf.float32), average_across_timesteps = False, average_across_batch = True)
loss = tf.reduce_sum(xEnt)
train = tf.train.AdamOptimizer(.001).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
	
for _ in range(num_epochs):
	avg_loss = 0
	loss_count = 0
	x, y = batched(tracks_train, batch_size, window_size)
	for i in range(len(x)):
		inpts = x[i]
		answrs = y[i]
		_, loss_val = sess.run([train, loss], feed_dict = {inpt: inpts, answr: answrs})
		avg_loss += loss_val
		loss_count += 1
	avg_loss /= float(loss_count)
	print(avg_loss)
		
print('Begin testing')
avg_loss = 0
loss_count = 0
x, y = batched(tracks_train, batch_size, window_size)
for i in range(len(x)):
	inpts = x[i]
	answrs = y[i]
	loss_val = sess.run(loss, feed_dict = {inpt: inpts, answr: answrs})
	avg_loss += loss_val
	loss_count += 1
avg_loss /= float(loss_count)
print(avg_loss)

print('Begin building')
state = x[0]
i = 0
while i < 100:
	inpts = state
	res = sess.run(logits, feed_dict = {inpt: inpts})
	state = [[np.argmax(res[j][k]) for k in range(len(state[j]))] for j in range(len(state))]
	print(segment_dict[inverse_indices[state[0][-1]]])
	i += 1