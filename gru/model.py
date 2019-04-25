import tensorflow as tf
import numpy as np
from Dice import dice
from Dice import parametric_relu as prelu 
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.python.ops.rnn import dynamic_rnn
from model_aux import *
np.random.seed(7)

class Model(object):
	def __init__(self, user_count, item_count, cate_count, embedding_size, hidden_size, attention_size, use_negsampling=False, travel_network=None, alpha=1e-4):
		with tf.variable_scope('inputs'):
			self.u = tf.placeholder(tf.int32, [None,], name='user_batch_id')
			self.i = tf.placeholder(tf.int32, [None,], name='item_batch_id')
			self.cate = tf.placeholder(tf.int32, [None,], name='category_batch_id')
			self.y = tf.placeholder(tf.float32, [None,], name='label_batch')
			self.hist_i = tf.placeholder(tf.int32, [None, None], name='item_hist')
			self.hist_cate = tf.placeholder(tf.int32, [None, None], name='category_hist')
			
			self.sl = tf.placeholder(tf.int32, [None,], name='seq_len_hist')
			self.lr = tf.placeholder(tf.float64, [])
			self.embedding_size = embedding_size
			
			self.reg_alpha = tf.constant(alpha, dtype=tf.float32, name='reg_alpha')
			self.use_negsampling = use_negsampling
			
			self.mask = tf.placeholder(tf.float32, [None, None], name='mask')

			self.embedding_size = embedding_size
			self.hidden_size = hidden_size
			self.attention_size = attention_size
			
		'''
			if use negative sampling during the training process,
			generate n items ids from negative sampling
		'''
		if self.use_negsampling:
			self.neg_hist_i = tf.placeholder(tf.int32, [None, None], name='neg_hist_item')
			self.neg_hist_cate = tf.placeholder(tf.int32, [None, None], name='neg_hist_category')

				
		# pre-trained embedding feeding
		self.embedding_placeholder = tf.placeholder(tf.float32, [item_count, self.embedding_size])
		with tf.variable_scope('embedding_layer') as scope:
			self.item_emb_w = tf.get_variable("item_embedding_var", [item_count, self.embedding_size])
			# assign pre-trained embedding to item_embeding_var
			self.embedding_init = self.item_emb_w.assign(self.embedding_placeholder)
			tf.summary.histogram('item_embedding_var', self.item_emb_w)
			self.item_emb_batch = tf.nn.embedding_lookup(self.item_emb_w, self.i)
			self.item_hist_embedding = tf.nn.embedding_lookup(self.item_emb_w, self.hist_i)
			if self.use_negsampling:
				self.neg_item_hist_embedding = tf.nn.embedding_lookup(self.item_emb_w, self.neg_hist_i)

			self.cate_emb_w = tf.get_variable("category_embedding_var", [cate_count, self.embedding_size])
			tf.summary.histogram('category_embedding_var', self.cate_emb_w)
			self.cate_emb_batch = tf.nn.embedding_lookup(self.cate_emb_w, self.cate)
			self.cate_hist_embedding = tf.nn.embedding_lookup(self.cate_emb_w, self.hist_cate)
			if self.use_negsampling:
				self.neg_cate_hist_embedding = tf.nn.embedding_lookup(self.cate_emb_w, self.neg_hist_cate)
				
			self.user_emb_w = tf.get_variable("user_embedding_var", [user_count, self.embedding_size])
			tf.summary.histogram('user_embedding_var', self.user_emb_w)
			self.user_emb_batch = tf.nn.embedding_lookup(self.user_emb_w, self.u)
			

		'''
			feeding item network
		'''
		with tf.variable_scope('travel_network'):
			self.item_network = tf.constant(travel_network, dtype=tf.int32, shape=[item_count, item_count])
			self.item_non_zeros = tf.count_nonzero(self.item_network, [0, 1])
		
			self.item_non_zero = tf.stop_gradient(tf.cast(tf.count_nonzero(self.item_network), tf.float32))
		
		'''
			item embeding with or without categorical information
		'''
		# --- with categorical embedding ---
		
		# self.item_eb = tf.concat([self.item_emb_batch, self.cate_emb_batch], -1)
		# self.item_hist_eb = tf.concat([self.item_hist_embedding, self.cate_hist_embedding], -1)
		# --- without categorical embedding ---
		self.item_eb = self.item_emb_batch
		self.item_hist_eb = self.item_hist_embedding
		self.item_hist_eb_sum = tf.reduce_sum(self.item_hist_eb, 1)
		
		if self.use_negsampling:
			# self.neg_hist_eb = tf.concat([self.neg_item_hist, self.neg_cate_hist], -1)
			self.neg_hist_eb = self.neg_item_hist_embedding
			self.neg_hist_eb_sum = tf.reduce_sum(self.neg_hist_eb, 1)
	'''
		essential part for fcn network layer
	'''
	def fcn_net(self, input, use_dice=False):
		with tf.variable_scope('fcn', reuse=tf.AUTO_REUSE) as scope:
			# bn1 = tf.layers.batch_normalization(inputs=input, name='bn1')
			bn1 = input
			dnn1 = tf.layers.dense(bn1, units=2*self.embedding_size, activation=None, name='f1')
			if use_dice:
				dnn1 = dice(dnn1, name='dice_1')
			else:
				dnn1 = prelu(dnn1, name='prelu1')
			
			dnn2 = tf.layers.dense(dnn1, 40, activation=None, name='f2')
			if use_dice:
				dnn2 = dice(dnn2, name='dice2')
			else:
				dnn2 = prelu(dnn2, name='prelu2')
			
			dnn3 = tf.layers.dense(dnn2, 1, activation=None, name='f3')
		return dnn3
	'''
		fully connected layer, calculating the possibility of
		target item being chosen.
		input concatenate the outputs of the gru model with the target item embedding
	'''
	def build_fcn_net(self, input, use_dice=False):

		self.y_hat = self.fcn_net(input, use_dice)
		self.y_hat = tf.reshape(self.y_hat, [-1])
		with tf.variable_scope('Metrics'):
			#cross_entropy_loss
			ctr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
				labels=self.y,
				logits=self.y_hat,
				name='cross_entropy'
			))
			self.loss = ctr_loss
			if self.use_negsampling:
				self.loss += self.aux_loss
			tf.summary.scalar('loss', self.loss)

			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(self.y_hat)), self.y), tf.float32), axis=-1)
			tf.summary.scalar('accuracy', self.accuracy)
		
		self.merged = tf.summary.merge_all()
	
	def build_fcn_net_sub(self, input, use_dice=False):
		self.y_hat_sub = self.fcn_net(input, use_dice)
		self.y_hat_sub = tf.reshape(self.y_hat_sub, [-1, tf.shape(self.y_hat_sub)[1]])
		self.y_hat_sub = tf.nn.softmax(self.y_hat_sub, axis=-1, name='probability')
	
	'''
		calculat the auxilary loss for negative sampling strategy for each timestamp
	'''
	def auxiliary_loss(self, h_states, pos_seq_eb, neg_seq_eb, mask, name='None'):
		mask = tf.cast(mask, tf.float32)
		pos_input = tf.concat([h_states, pos_seq_eb], -1)
		neg_input = tf.concat([h_states, neg_seq_eb], -1)
		
		pos_input_prob = self.auxiliary_net(pos_input)
		neg_input_prob = self.auxiliary_net(neg_input)
		'''
			only consider the mask prob for each postive/negative sequence
			mask indicates the lenght for each sequence in tensor's form
		'''
		aux_pos_loss = -tf.reshape(tf.log(pos_input_prob), [-1, tf.shape(pos_seq_eb)[1]]) * mask
		aux_neg_loss = -tf.reshape(tf.log(1-neg_input_prob), [-1, tf.shape(neg_seq_eb)[1]]) * mask
		aux_loss = tf.reduce_mean(tf.reduce_sum(aux_pos_loss+aux_neg_loss, -1), 0)
		
		return aux_loss
	
	'''
		auxiliary network with positive input and negative sampling input
		calculate the probs for each inputs with current history embedding seperatly.
	'''
	def auxiliary_net(self, input, name='aux_net'):
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
			# bn1 = tf.layers.batch_normalization(inputs=input, name='b1')
			bn1 = input
			dnn1 = tf.layers.dense(bn1, self.embedding_size, activation=tf.nn.sigmoid, name='f1')
			dnn2 = tf.layers.dense(dnn1, 40, activation=tf.nn.sigmoid, name='f2')
			dnn3 = tf.layers.dense(dnn2, 1, activation=tf.nn.sigmoid, name='f3')

			y_hat = dnn3
			return y_hat
	'''
		traditional training process, 
	'''
	def train(self, sess, inputs, lr):
		if self.use_negsampling:
			loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer], feed_dict={
				self.lr: lr,
				self.u: inputs[0],
				self.i: inputs[1],
				self.cate: inputs[2],
				self.hist_i: inputs[3],
				self.hist_cate: inputs[4],
				self.mask: inputs[5],
				self.y: inputs[6],
				self.sl: inputs[7],
				self.neg_hist_i: inputs[8],
				self.neg_hist_cate: inputs[9]
			})

			return loss, accuracy, aux_loss
		else:
			loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
				self.lr: lr,
				self.u: inputs[0],
				self.i: inputs[1],
				self.cate: inputs[2],
				self.hist_i: inputs[3],
				self.hist_cate: inputs[4],
				self.mask: inputs[5],
				self.y: inputs[6],
				self.sl: inputs[7],
			})
			return loss, accuracy, 0
	'''
		only calculat the output for each inputs, no training process, no 
		weight will be updated.
		(consider this as testing f)
	'''
	def calculate(self, sess, inputs, lr):
		if self.use_negsampling:
			probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict={
				self.lr: lr,
				self.u: inputs[0],
				self.i: inputs[1],
				self.cate: inputs[2],
				self.hist_i: inputs[3],
				self.hist_cate: inputs[4],
				self.mask: inputs[5],
				self.y: inputs[6],
				self.sl: inputs[7],
				self.neg_hist_i: inputs[8],
				self.neg_hist_cate: inputs[9]
			})

			return probs, loss, accuracy, aux_loss
		else:
			probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
				self.lr: lr,
				self.u: inputs[0],
				self.i: inputs[1],
				self.cate: inputs[2],
				self.hist_i: inputs[3],
				self.hist_cate: inputs[4],
				self.mask: inputs[5],
				self.y: inputs[6],
				self.sl: inputs[7],
			})
			return probs, loss, accuracy, 0
	
	def calculate_multi(self, sess, inputs, lr):
		
		if self.use_negsampling:
			probs, loss, aux_loss = sess.run([self.y_hat_sub, self.loss, self.aux_loss], feed_dict={
				self.lr: lr,
				self.u:	inputs[0],
				self.i:	inputs[1],
				self.cate:	inputs[2],
				self.hist_i:	inputs[3],
				self.hist_cate:	inputs[4],
				self.mask:	inputs[5],
				self.y:	inputs[6],
				self.sl: inputs[7],
				self.neg_hist_i: inputs[8],
				self.neg_hist_cate: inputs[9]
			})
			return probs, loss, aux_loss
		else:
			probs, loss = sess.run([self.y_hat_sub, self.loss], feed_dict={
				self.lr: lr,
				self.u:	inputs[0],
				self.i:	inputs[1],
				self.cate:	inputs[2],
				self.hist_i:	inputs[3],
				self.hist_cate:	inputs[4],
				self.mask:	inputs[5],
				self.y:	inputs[6],
				self.sl: inputs[7],
			})
			return probs, loss, 0

	def save(self, sess, path):
		saver = tf.train.Saver()
		saver.save(sess, save_path=path)

	def restrore(self, sess, path):
		saver = tf.train.Saver()
		saver.retore(sess, save_path=path)
		print("restore model from %s" % path)

	def pre_train(self, sess, embedding):
		_ = sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embedding})
		print('loading the pre_train data')
		return


'''
	gru - attention - gru
'''

class Model_GRU_ATT_GRU(Model):
	def __init__(self, user_count, item_count, cate_count, attention_size, hidden_units, use_negsampling=False, embedding_size=64, travel_network=None, alpha=1e-4):
		super(Model_GRU_ATT_GRU, self).__init__(user_count, item_count, cate_count, embedding_size, hidden_units,  attention_size, use_negsampling, travel_network, alpha)

		'''
			RNN layers
		'''
		with tf.variable_scope('rnn_1'):
			rnn_outputs, _ = dynamic_rnn(
				GRUCell(hidden_units), inputs=self.item_hist_eb,
				sequence_length=self.sl, dtype=tf.float32,
				scope='gru1'
			)
			tf.summary.histogram('GRU outputs', rnn_outputs)

		aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_hist_eb[:, 1:, :],
										self.neg_hist_eb[:, :-1, :], self.mask[:, 1:], name="gru")
		self.aux_loss = aux_loss_1
		'''
			Attention layer, att_outputs will be the weighted one w.r.t. given target item.
		'''
		with tf.variable_scope('attention_layer1'):
			att_outputs, weights = fcn_attention(self.item_eb, rnn_outputs, attention_size, self.mask,
												use_softmax=True, name='att_1', mode='list', return_weight=True)

			tf.summary.histogram('weight_outputs', weights)
		
		'''
			Attention layer for multiple target, reuse same attention block as one target version.
		'''
		with tf.variable_scope('attention_layer1', reuse=True):
			
			queries = tf.tile(tf.expand_dims(self.item_emb_w, 0), [tf.shape(self.item_eb)[0], 1, 1])
			att_outputs_sub, weights_sub = fcn_attention_multi(queries, rnn_outputs, attention_size, self.mask, 
												use_softmax=True, name='att_1', mode='list', return_weight=True)
			tf.summary.histogram('weight_outputs_sub', weights_sub)


		with tf.variable_scope('rnn_2'):
			rnn_outputs2, final_states2 = dynamic_rnn(
				GRUCell(hidden_units), inputs=att_outputs, 
				sequence_length=self.sl, dtype=tf.float32, scope='gru2')
			tf.summary.histogram('GRU2_final_states', final_states2)
		
		'''
			GRU layer for input with [B, N, T, H] shape
		'''
		sl_sub = tf.expand_dims(self.sl, 1)
		sl_sub = tf.tile(sl_sub, [1, item_count])
		sl_sub = tf.reshape(sl_sub, [-1])

		att_outputs_sub = tf.reshape(att_outputs_sub, [-1, tf.shape(self.item_hist_eb)[1], hidden_units])

		with tf.variable_scope('rnn_2', reuse=True):
			rnn_output_sub, final_states_sub = dynamic_rnn(
				GRUCell(hidden_units), inputs=att_outputs_sub,
				sequence_length=sl_sub, dtype=tf.float32, scope='gru2'
			)
			
			final_states_sub = tf.reshape(final_states_sub, [tf.shape(self.item_eb)[0], item_count, hidden_units])
			tf.summary.histogram('GRU2_final_states_sub', final_states_sub)
		# input = tf.concat([self.user_emb_batch, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_states2], -1)
		'''
			fully connected layer for calculate the probability
		'''
		item_eb_sub = tf.expand_dims(self.item_emb_w, 0)
		item_eb_sub = tf.tile(item_eb_sub, [tf.shape(self.item_eb)[0], 1, 1])
		
		item_hist_eb_sum_sub = tf.expand_dims(self.item_hist_eb_sum, 1)
		item_hist_eb_sum_sub = tf.tile(item_hist_eb_sum_sub, [1, item_count, 1])
		
		
		# input = tf.concat([self.item_eb, self.item_hist_eb_sum, self.item_eb * self.item_hist_eb_sum, final_states2], -1)
		# input_sub = tf.concat([item_eb_sub, item_hist_eb_sum_sub, item_eb_sub * item_hist_eb_sum_sub, final_states_sub], -1)
		input = tf.concat([self.item_eb, final_states2], -1)
		input_sub = tf.concat([item_eb_sub, final_states_sub], -1)
		self.build_fcn_net(input, use_dice=True)
		self.build_fcn_net_sub(input_sub, use_dice=True)



'''
	different attention strategy w.r.t above model
	using stacked gru model before add the attention layer
'''

class Model_GRU_GRU_ATT(Model):
	def __init__(self, user_count, item_count, cate_count, attention_size, hidden_units, use_negsampling=False, embedding_size=64, travel_network=None, alpha=1e-4):
		super(Model_GRU_GRU_ATT, self).__init__(user_count, item_count, embedding_size, hidden_units,  attention_size, use_negsampling, travel_network, alpha)

		'''
			rnn layers
		'''
		with tf.variable_scope('rnn_1'):
			rnn_outputs, _ = dynamic_rnn(
				GRUCell(hidden_units), inputs=self.item_hist_eb,
				sequence_length=self.sl, dtype=tf.float32,
				scope='gru1'
			)
			tf.summary.histogram('GRU_outputs', rnn_outputs)

		aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_hist_eb[:, 1:, :],
													self.neg_hist_eb[:, :-1, :], self.mask[:, 1:], name="gru")
		self.aux_loss = aux_loss_1
		with tf.variable_scope('rnn_2'):
			rnn_outputs2, _ = dynamic_rnn(
				GRUCell(hidden_units), inputs=rnn_outputs,
				sequence_length=self.sl, dtype=tf.float32,
				scope='gru2'
			)
			tf.summary.histogram('GRU_outputs2', rnn_outputs2)	

		
		with tf.variable_scope('attention_layer1'):
			att_feature, weights = fcn_attention(self.item_eb, rnn_outputs2, attention_size, 
												self.mask, use_softmax=True, name='att_1', mode='sum', return_weight=True)
			# '''
			# 	using reduce_sum performs the same as the sum mode
			# '''
			# att_feature = tf.reduce_sum(att_outputs, 1)
			tf.summary.histogram('attention_feature', att_feature)
		
		with tf.variable_scope('attention_layer1', reuse=tf.AUTO_REUSE):
			queries = tf.tile(tf.expand_dims(self.item_emb_w, 0), [item_count, 1, 1])
			att_feature_sub, weights_sub = fcn_attention_multi(queries, rnn_outputs2, attention_size, 
																self.mask, use_softmax=True, name='att_1', mode='sum', return_weight=True)	
		# input = tf.concat(
		# 	[self.user_emb_batch, self.item_emb_batch, self.item_his_eb_sum,
		# 	 self.item_eb*self.item_his_eb_sum, att_feature], 1)

		item_eb_sub = tf.expand(self.item_emb_w, 0)
		item_eb_sub = tf.tile(item_eb_sub, [tf.shape(self.item_eb)[0], 1, 1])

		item_hist_eb_sum_sub = tf.expand_dims(self.item_hist_eb_sum, 1)
		item_hist_eb_sum_sub = tf.tile(item_hist_eb_sum_sub, [1, item_count, 1])

		# input = tf.concat([self.item_eb, self.item_hist_eb_sum, self.item_eb*self.item_hist_eb_sum, att_feature], -1)
		# input_sub = tf.concat([item_eb_sub, item_hist_eb_sum_sub, item_eb_sub * item_hist_eb_sum_sub, att_feature_sub], -1)
		
		input = tf.concat([self.item_eb, att_feature], -1)
		input_sub = tf.concat([item_eb_sub, att_feature_sub], -1)
		
		self.build_fcn_net(input, use_dice=True)
		self.build_fcn_net_sub(input_sub, use_dice=True)
'''
	simple gru model without attention mechanism
'''
class Model_GRU(Model):
	def __init__(self, user_count, item_count, cate_count, hidden_units, use_negsampling=False, embedding_size=64, travel_network=None, alpha=1e-4):
		super(Model_GRU, self).__init__(user_count, item_count, cate_count, embedding_size, hidden_units,  None, use_negsampling, travel_network, alpha)
		'''
			simple rnn model
		'''
		with tf.variable_scope('rnn_1'):
			rnn_outputs, final_states = dynamic_rnn(
				GRUCell(hidden_units), inputs=self.item_hist_eb,
				sequence_length=self.sl, dtype=tf.float32,
				scope='gru1'
			)
			tf.summary.histogram('GRU_final_states', final_states)

		aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_hist_eb[:, 1:, :],
										self.neg_hist_eb[:, :-1, :], self.mask[:, 1:], name="gru")
		self.aux_loss = aux_loss_1
		# input = tf.concat([
		# 	self.user_emb_batch, self.item_emb_batch, self.item_his_eb_sum,
		# 	self.item_eb*self.item_his_eb_sum, final_states
		# ], -1)

		item_eb_sub = tf.expand_dims(self.item_emb_w, 0)
		item_eb_sub = tf.tile(item_eb_sub, [tf.shape(self.item_emb_batch)[0], 1, 1])

		item_hist_eb_sum_sub = tf.expand_dims(self.item_hist_eb_sum, 1)
		item_hist_eb_sum_sub = tf.tile(item_hist_eb_sum_sub, [1, item_count, 1])

		final_states_sub = tf.expand_dims(final_states, 1)
		final_states_sub = tf.tile(final_states_sub, [1, item_count, 1])

		input = tf.concat([self.item_eb, final_states], -1)
		input_sub = tf.concat([item_eb_sub, final_states_sub], -1)
		
		self.build_fcn_net(input, use_dice=True)
		self.build_fcn_net_sub(input_sub, use_dice=True)

'''
	simple gru model with 2 stacked version
'''
class Model_GRU_GRU(Model):
	def __init__(self, user_count, item_count, cate_count, hidden_units, use_negsampling=False, embedding_size=64, travel_network=None, alpha=0.0001):
		super(Model_GRU_GRU, self).__init__(user_count, item_count, cate_count, embedding_size, hidden_units,  None, use_negsampling, travel_network, alpha)

		'''
			simple rnn_1 model 
		'''	
		with tf.variable_scope('rnn_1'):
			rnn_outputs, _ = dynamic_rnn(
				GRUCell(hidden_units), inputs=self.item_hist_eb,
				sequence_length=self.sl, dtype=tf.float32,
				scope='gru1'
			)
			tf.summary.histogram('GRU_outputs', rnn_outputs)
		
		aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_hist_eb[:, 1:, :],
										self.neg_hist_eb[:, :-1, :], self.mask[:, 1:], name="gru")
		self.aux_loss = aux_loss_1

		with tf.variable_scope('rnn_2'):
			rnn_outputs2, final_states2 = dynamic_rnn(
				GRUCell(hidden_units), inputs=rnn_outputs,
				sequence_length=self.sl, dtype=tf.float32,
				scope='gru2'
			)
		
			tf.summary.histogram('GRU_final_states', final_states2)

		
		item_eb_sub = tf.expand_dims(self.item_emb_w, 0)
		item_eb_sub = tf.tile(item_eb_sub, [tf.shape(self.item_eb)[0], 1, 1])

		item_hist_eb_sum_sub = tf.expand_dims(self.item_hist_eb_sum, 1)
		item_hist_eb_sum_sub = tf.tile(item_hist_eb_sum_sub, [1, item_count, 1])

		final_states2_sub = tf.expand_dims(final_states2, 1)
		final_states2_sub = tf.tile(final_states2_sub, [1, item_count, 1])
		
		# input = tf.concate([self.item_eb, self.item_hist_eb_sum, self.item_eb*self.item_hist_eb_sum, final_states2], -1)
		# input_sub = tf.concat([item_eb_sub, item_hist_eb_sum_sub, item_eb_sub*item_hist_eb_sum_sub, final_states2_sub], -1)
		
		input = tf.concat([self.item_eb, final_states2], -1)
		input_sub = tf.concat([item_eb_sub, final_states2_sub], -1)
		
		self.build_fcn_net(input, use_dice=True)
		self.build_fcn_net_sub(input_sub, use_dice=True)
	

class Model_GRU_ATT(Model):
	def __init__(self, user_count, item_count, cate_count, attention_size, hidden_units, use_negsampling=False, embedding_size=64, travel_network=None, alpha=0.0001):
		super(Model_GRU_ATT, self).__init__(user_count, item_count, cate_count, embedding_size, hidden_units,  attention_size, use_negsampling, travel_network, alpha)
		'''
			simple dynmaic rnn module
		'''
		with tf.variable_scope('rnn1'):
			rnn_outputs, _ = tf.nn.dynamic_rnn(
				GRUCell(hidden_units), inputs=self.item_hist_eb,
				sequence_length=self.sl, dtype=tf.float32,
				scope='gru1'
			)
		aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_hist_eb[:, 1:, :],
										self.neg_hist_eb[:, :-1, :], self.mask[:, 1:], name="gru")

		self.aux_loss =  aux_loss_1
		# indice = self.sl - tf.ones_like(self.sl)
		range = tf.expand_dims(tf.range(tf.shape(self.sl)[0]), 1)
		pos = tf.expand_dims(tf.subtract(self.sl, tf.ones_like(self.sl)), 1)
		indices = tf.concat([range, pos], -1)

		last_item_indices = tf.gather_nd(self.hist_i, indices)

		queries = tf.nn.embedding_lookup(self.item_emb_w, last_item_indices)
		# queries = tf.tile(tf.expand_dims(queries, 1), [1, tf.shape(self.hist_i)[1], 1])

		'''
			Attention layer, att_outputs will be the weighted one w.r.t. given target item.
		'''
		with tf.variable_scope('attention_layer1'):
			att_outputs, weights = fcn_attention(queries, rnn_outputs, attention_size, self.mask,
												use_softmax=True, name='att_1', mode='sum', return_weight=True)

			tf.summary.histogram('weight_outputs', weights)
			print(att_outputs.get_shape().as_list())
		att_outputs_sub =  tf.tile(tf.expand_dims(att_outputs, 1), [1, item_count, 1])
		item_eb_sub = tf.tile(tf.expand_dims(self.item_emb_w, 0), [tf.shape(self.item_eb)[0], 1, 1])
		# input = tf.concat([att_outputs, self.item_eb, att_outputs-self.item_eb, att_outputs*self.item_eb])
		input = tf.concat([self.item_eb, att_outputs], -1)
		input_sub = tf.concat([item_eb_sub, att_outputs_sub], -1)

		self.build_fcn_net(input, use_dice=True)
		self.build_fcn_net_sub(input_sub, use_dice=True)