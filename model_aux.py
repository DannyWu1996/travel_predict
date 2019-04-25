import tensorflow as tf
from Dice import parametric_relu as prelu
from Dice import dice
from data_iterator import *
from scipy.sparse import csr_matrix

'''
	attention mechanism for multiple target
'''
def fcn_attention_multi(queries, facts, attention_size, mask, name='null',  mode='sum', use_softmax=True, time_major=False, return_weight=False, forCnn=False):
	'''
		queries:	[B, N, H1] N is the number of the item 
		facts:		[B, T, H2] T is the longest length of the history
		mask:		[B, T]
	'''
	if isinstance(facts, tuple):
		'''
			in case for bi-direction rnn, concate the frontier and back rnn outputs
		'''
		tf.concat(facts, -1)
	
	if len(facts.get_shape().as_list())==2:
		facts = tf.expand_dims(facts, 1)

	if time_major:
		facts = tf.transpose(facts, [1, 0, 2])
	
	# turn mask into boolean tensor
	mask = tf.equal(mask, tf.ones_like(mask))
	mask = tf.expand_dims(mask, 1)
	mask = tf.tile(mask, [1, tf.shape(queries)[1], 1])

	fact_size = facts.get_shape().as_list()[-1]
	query_size = queries.get_shape().as_list()[-1]

	with tf.variable_scope(name, reuse=True):
		queries = tf.layers.dense(queries, fact_size, activation=None, name='f1')
		queries = dice(queries)

		queries = tf.expand_dims(queries, 2)
		queries = tf.tile(queries, [1, 1, tf.shape(facts)[1], 1])

		facts = tf.expand_dims(facts, 1)
		facts = tf.tile(facts, [1, tf.shape(queries)[1], 1, 1])

		input_all = tf.concat([queries, facts, queries-facts, queries*facts], -1)
		d_layer_1_all = tf.layers.dense(input_all, 40, activation=tf.nn.sigmoid, name='f1_att')
		d_layer_2_all = tf.layers.dense(d_layer_1_all, 20, activation=tf.nn.sigmoid, name='f2_att')
		d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')
		
		# reshape scores as [B, N, 1, T]
		scores = tf.transpose(d_layer_3_all, [0, 1, 3, 2]) 

	key_masks = tf.cast(tf.expand_dims(mask, 2), tf.bool)
	paddings = tf.ones_like(scores) * (-2**32+1)

	if not forCnn:
		scores = tf.where(key_masks, scores, paddings)
	
	if use_softmax:
		scores = tf.nn.softmax(scores, axis=-1)
	
	if mode == 'sum':
		output = tf.matmul(scores, facts)
		output_shape = tf.gather(tf.shape(facts), [0,1,3])	
		output = tf.reshape(output, output_shape)
	else:
		scores = tf.transpose(scores, [0, 1, 3, 2])
		output = facts * scores
	if return_weight:
		return output, scores
	return output, 0
'''
	attention mechanism for single one
'''
def fcn_attention(query, facts, attention_size, mask, name='null', mode='sum', use_softmax=True, time_major=False,  return_weight=False, forCnn=False):
	if isinstance(facts, tuple):
		'''
			in case for bi-direction rnn, concate the frontier and back rnn outputs
		'''
		tf.concat(facts, -1)
	if len(facts.get_shape().as_list())==2:
		facts = tf.expand_dims(facts, 1)

	if time_major:
		'''
			convert (T, B, D) => (B, T, D)
		'''
		facts = tf.transpose(facts, [1, 0, 2])
	 
	'''
		trainable parameters
	'''
	#	create boolean size mask
	mask = tf.equal(mask, tf.ones_like(mask))
	#   get hidden units of rnn output
	facts_size = facts.get_shape().as_list()[-1]
	query_size = query.get_shape().as_list()[-1]
	with tf.variable_scope(name):
		'''
			convert query into same dimention as rnn outputs
			by using single fcn
		'''
		query = tf.layers.dense(query, facts_size, activation=None, name='f1')
		query = dice(query)

		'''
			query is only 2 dimentional array, (T, D).
			tile queries as (T, D*B),
			reshape queries same as facts, (T, B, D)
		'''
		queries = tf.tile(query, [1, tf.shape(facts)[1]])
		queries = tf.reshape(queries, tf.shape(facts))

		'''
			concatenate all the element into one single input, shape(T, B, X), where X is the total dimention of all the embedidng
		'''
		input_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
		d_layer_1_all = tf.layers.dense(input_all, 40, activation=tf.nn.sigmoid, name='f1_att')
		d_layer_2_all = tf.layers.dense(d_layer_1_all, 20, activation=tf.nn.sigmoid, name='f2_att')
		d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')
		'''
			fcn layer to calcualte attention value for each time stamp's rnn output 
			w.r.t corresponding query item
			(B, 1, T)
		'''
		# d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
		d_layer_3_all = tf.transpose(d_layer_3_all, [0, 2, 1])
		scores = d_layer_3_all

		'''
			mask and paddings
			paddings with -2**32 indicates zero while using sigmoid activation
			mask: (T, 1, B)
		'''
		key_masks = tf.expand_dims(mask, 1)
		paddings = tf.ones_like(scores) * (-2**32+1)

		if not forCnn:
			'''
				select scores based on the key_mask, 
				real score when key_mask is True, negative infinity otherwise.
			'''
			scores = tf.where(key_masks, scores, paddings)

		if use_softmax:	
			'''
				determine whether using softmax to normalize all score as probability.
				by using softmax, scores with negative infinity will be 0.
			'''
			scores = tf.nn.softmax(scores, axis=-1)
			
		'''
			pooling strategy, weighted summation
		'''
		if mode == 'sum':
			output = tf.matmul(scores, facts)
			output = tf.reshape(output, [-1, facts_size])
		else:
			'''
				scores: (T, B), generate a tensor with same shape as inputs, 
				but the weighted version.
			'''
			# scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
			# output = facts * tf.expand_dims(scores, -1)
			scores = tf.transpose(scores, [0, 2, 1])
			output = facts * scores
			output = tf.reshape(output, tf.shape(facts))

		if return_weight:
			return output, scores
		return output, 0

'''
	evaluation metric
'''
def get_top_k_recommendation(score_arr, current_citys, item_count, city_network=None):
  # ind = np.argpartition(score_arr, -top_k)[:, -top_k:]
	if city_network is None:
		city_network = np.ones([item_count, item_count]) - np.eye(item_count, dtype=int)
		city_network = csr_matrix(city_network)
	top_k_arr = []
	for i in range(score_arr.shape[0]):
		relate_city = city_network[current_citys[i]].indices
		# rec_city = np.zeros(top_k)
        # if len(relate_city) <= top_k:
        #     ind = np.argsort(np.take(score_arr[i], relate_city))[::-1]
        #     top_k_arr.append(np.hstack((relate_city[ind], np.zeros(top_k-len(relate_city)))))
        # else:
		score = np.take(score_arr[i], relate_city)
		ind =  np.argsort(score)
		top_k_arr.append(relate_city[ind[::-1]])
		# ind = np.argpartition(score, -top_k)[-top_k:]
		# top_k_arr.append(relate_city[ind[np.argsort(np.take(score, ind))][::-1]])
	return np.stack(top_k_arr, axis=0).reshape((-1, item_count-1))

def get_recall_MRR_k(pred_labels, true_labels, top_k):
	hit = 0
	rank = 0
	correct_set = set()
	rank_correct = 0
	for i, true_label in enumerate(true_labels):
		# if true_labels[i] in pred_labels_k[i]:
		#     hit += 1
		index = pred_labels[i].tolist().index(true_label)
		if index<top_k:
			hit+=1
			correct_set.add(true_label)
			rank_correct += index+1
		rank += index+1	
	print('correct set is {}, MMR for correct set is {}'.format(correct_set, float(rank_correct)/hit))
	return float(hit) / len(true_labels), float(rank) / len(true_labels)

def get_precision_k(pred_labels_k, true_labels, top_k, user_num):
    hit = 0
    for i in range(len(true_labels)):
        if true_labels[i] in pred_labels_k[i]:
            hit += 1
    return float(hit) / (user_num * top_k)

