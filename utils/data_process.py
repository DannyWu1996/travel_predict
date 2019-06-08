import sys
import random
import time
import re
import pandas as pd
from scipy.sparse import csr_matrix
import pickle
import os
import numpy as np
import argparse

# %load ./utils/do_preprocess.py
'''
Create on Jan 7， 2019
@author: danny.W
'''

random.seed(1234)
def json_to_df(file_path):
	"""
	:param file_path: json file path
	:return: Pandas.Dataframe object
	"""

	#check whether file path is exist
	assert os.path.exists(file_path), "file unknown!!!"
	with open(file_path, 'r') as fin:
		dic = {}
		indx = 0
		overlapped = 0
		# iterative read file line by line
		for line in fin:
			# convert string expression line into dict format
			record = eval(line)
			# check overlapped record if departure city is same as the destination city
			if(record['dpt']==record['dst']):
				overlapped += 1
				continue
			dic[indx] = eval(line)
			indx += 1
		# convert dict format into pandas dataframe
		df = pd.DataFrame.from_dict(dic, orient='index')
		df.reset_index(drop=True)
		print("delete overlapped record %d" %  overlapped)
		return df

def save_df(file_path, df_object):
	"""
	saving pandas' dataframe into pickle file format
	:param file_path: file path to save the dataframe object
	:param df_object: pandas.Dataframe object, input dataframe object to save
	"""
	with open(file_path, 'wb') as f:
		pickle.dump(df_object, f)
	print("\tsaving dataframe to path - %s" % file_path)


def single_df_set(df, col_name):
	"""
	find the unique value set for the target column in dataframe
	:param df: target dataframe
	:param col_name: target column name
	:return: unique value set for target column
	"""
	if col_name == 'item':
		# concatenate the value with both departure city and arrival city
		city_list = np.unique(np.concatenate(
			(df['dpt'].unique(), df['dst'].unique()),
			0
		)).tolist()
		return city_list
	else:
		key_set = np.unique(df[col_name]).tolist()
		return key_set

def build_map(df, dropped_df, col_name):
	"""
	build a mapping from string to int
	:param df: Pandas.Dataframe,	target object
	:param dropped_df:	Pandas.Dataframe	dropped object (if overlapped)
	:param col_name: string, target column name
	:return: (map, key) tuple
	"""
	# print df.columns.values
	if col_name == 'item':
		# concatenate both departure city and arrival city
		city_list = np.unique(np.concatenate(
			(df['dpt'].unique(),
			 df['dst'].unique(),
			 dropped_df['dpt'].unique(),
			 dropped_df['dst'].unique()
			 ), 0)).tolist()
		# print city_list
		key = sorted(city_list)
		m = dict(zip(key, range(len(key))))
		# replace the departure city by id
		df['dpt'] = df['dpt'].map(lambda x: m[x])
		dropped_df['dpt'] = dropped_df['dpt'].map(lambda x: m[x])
		# repalce the arrival city by id
		df['dst'] = df['dst'].map(lambda x: m[x])
		dropped_df['dst'] = dropped_df['dst'].map(lambda x: m[x])
	else:
		key = sorted(np.unique(np.concatenate(
			(df[col_name].unique(),
			dropped_df[col_name].unique()
			), 0)).tolist())
		m = dict(zip(key, range(len(key))))
		df[col_name] = df[col_name].map(lambda x: m[x])
		dropped_df[col_name] = dropped_df[col_name].map(lambda x: m[x])
	return m, key


def do_split_data(file_path, saved_file_path, city_list):
	"""
	split row data from row travel history into single travel records format
	:param file_path:	input raw tavel history file
	:param saved_file_path: output file path
	:param city_list: Domestic airport(city-wise) list
	:return:
	"""
	assert os.path.exists(file_path), "file unknown!!!"
	travel_history = []

	def remove_abroad(travel_series, valid_set):
		"""

		:param travel_series: travel history
		:param valid_set: Domestic airport(city-wise) list
		:return: valid series with all domestic airport (city)
		"""
		valid_series = []
		for loc in travel_series:
			if loc not in valid_set:
				continue
			valid_series.append(loc)
		return valid_series

	with open(file_path, 'r') as fin:
		for line in fin:
			# print line
			order_id, user_id, travel_series, time_series = line.split()
			travel_series = re.split('[/ -]', travel_series[1:-1])
			travel_series = remove_abroad(travel_series, city_list)
			travel_tuples = zip(
				travel_series[:-1], travel_series[1:], time_series[1:-1].split('-'))
			for tuple in travel_tuples:
				if '/' in tuple[2]:
					continue
				travel_history.append(
					{'user_id': user_id[1:-1], 'order_id': order_id[1:-1], 'dpt': tuple[0], 'dst': tuple[1], 'time': tuple[2]})

	# f_dir,  f_name = os.path.split(file_path)
	# json_file_path = os.path.join(f_dir, f_name.split('.')[0]+'.json')
	# print '\tsaving split data into - %s' % saved_file_path
	with open(saved_file_path, 'w+') as fout:
		fout.write('\n'.join([str(x) for x in travel_history]))


'''
	remove user whose total orders are less than minimum order numbers
'''


def remove_user_record(history_df, min_order):
	"""

	:param history_df: Pandas.Dataframe, user history dataframe object
	:param min_order: int, shreshold of minimum order
	:return: user history dataframe after above the min_order shreshold
	"""
	remove_records = []
	remove_user_nums = 0
	index = 0
	# sort dataframe by user_id
	history_df = history_df.sort_values('user_id')
	for user_id, hist in history_df.groupby('user_id'):
		# if current user's history is less than min_order, drop it
		if len(hist) < min_order:
			remove_user_nums += 1
			remove_records += [i for i in range(index, index+len(hist))]
		index += len(hist)

	print("\tremove user nums: %d" % remove_user_nums)
	# history dataframe of dropped user
	dropped_df = history_df.loc[history_df.index[remove_records]]
	# history dataframe of valid user
	history_df = history_df.drop(history_df.index[remove_records])

	u = np.unique(history_df['user_id'])
	dropped_u = np.unique(dropped_df['user_id'])
	# make sure the dropped users' set doesn't intersect with the
	# valid users' set
	if len(np.intersect1d(u, dropped_u))>0:
		raise BaseException("user intersection error!!!!")

	return history_df, dropped_df

def travel_net_gen(travel_net_df, item_map):
	"""

	:param travel_net_df: travel network dataframe
	:param item_map: city2id map
	:return: sparse matraix contains travel network
	"""
	airport_list = travel_net_df.columns.values[1:]
	selected = [item in item_map for item in airport_list]
	travel_net_df = travel_net_df.iloc[selected, [True]+selected]
	## map airport to id, then sort rows by their airport id, then remvoe the airport column
	travel_net_df['airport'] = travel_net_df['airport'].map(
		lambda x: item_map[x])
	travel_net_df = travel_net_df.sort_values('airport')
	travel_net_df = travel_net_df.drop('airport', 1)

	## map the column to id, then sort the column id.
	travel_net_df = travel_net_df.rename(index=str, columns=item_map)
	travel_net_df = travel_net_df.reindex(
		columns=sorted(travel_net_df.columns))
	## return sparse matrix for adjcency travel_route network
	sparse = csr_matrix(travel_net_df.values.T)
	print('all %d travel routes' % sparse.count_nonzero())
	return sparse


'''
	parameter setting for data_process
'''
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--preprocess", action='store_true',
					help="choose whether to process the data")
parser.add_argument("-s", "--do_split", action='store_true',
					help="Split raw json file to specific format. Format, uid, dpt, dst, order_id")
parser.add_argument("-r", "--raw_history_data_path", type=str,
					help="Path to raw history data. Json format, with column user_id, dpt, dst, time, order_id")
parser.add_argument("-m", "--min_order", type=int,
					help="Users who have less than \"min_order\" orders should be removed", default=0)
parser.add_argument("-a", "--aux_path", type=str,
					help="Path to processed data, including user2id, city2id mappings")
parser.add_argument("-u", "--raw_user_data_path", type=str,
					help="Path to raw user data, Json format, with columns user_id, age, gender, travel_miles", default='')
parser.add_argument("-l", "--long_seq", action="store_false", help='cutting strategy for long or short sequence')
parser.add_argument("-sr", "--split_ratio", type=float, help="user_based split ratio for test and training dataset", default=0.8)
parser.add_argument("--nonhybrid", action="store_true", help='using hybrid dataset trainset prefix')
args = parser.parse_args()

do_split = args.do_split
do_preprocess = args.preprocess
path_history = args.raw_history_data_path
min_order = args.min_order
aux_path = args.aux_path
path_user = args.raw_user_data_path
long_seq = args.long_seq
split_ratio = args.split_ratio
nonhybrid = args.nonhybrid
'''
	history-wised splitting strategy
'''

if do_preprocess:
	do_preprocess = args.preprocess
	path_history = args.raw_history_data_path
	aux_path = args.aux_path + '_%d' % min_order
	path_user = args.raw_user_data_path
	print("==========================data preprocessing setting==========================")
	print("\tsaving preprocessed aux path - %s" % aux_path)
	print("\traw history data path - %s" % path_history)
	if path_user != '':
		print("\traw user data path - %s" % path_user)
	print("\tconvert the raw data to json format - %s" %
		  ('True' if do_split is not None else 'False'))
	print("\tminimum order number - %d" % min_order)
	print("\tcutting long sequence - %s" % ("True" if long_seq else "False"))
	print("\tnonhybrid - %s" % ("True" if nonhybrid else "False"))
	print("==============================================================================")

	with open('../raw_data/itny/city_air_map.pkl', 'rb') as f:
		city_list, air_list, air_city_map = pickle.load(f, encoding='bytes')
	if do_split:
		# print do_split
		do_split_data(do_split, path_history, city_list)

	if not os.path.isdir(aux_path):
		os.mkdir(aux_path)

	# convert history with json format into pandas.dataframe object
	history_df = json_to_df(path_history)
	# split user according to min_order threshold
	history_df, dropped_df = remove_user_record(history_df, min_order)

	# build the mapping for each dataframe column (user_id, item, order_id)
	user_map, user_key = build_map(history_df, dropped_df, 'user_id')
	item_map, item_key = build_map(history_df, dropped_df, 'item')
	order_map, order_key = build_map(history_df, dropped_df, 'order_id')

	# create category list (same as item_map key if is not specified_
	cate_list = np.arange(len(item_map))

	# find unique set for giving volumn name (user_id, item, order_id_
	hist_user = single_df_set(history_df, 'user_id')
	hist_item = single_df_set(history_df, 'item')
	hist_order = single_df_set(history_df, 'order_id')

	# calculate user_count, order_count, category_count,
	# mostly by counting the length of the key of the corresponding map.
	user_count, item_count, order_count, cate_count =\
		len(hist_user), len(item_map), len(hist_order), len(cate_list)
	print('user_count: %d\titem_count: %d\torder_count: %d\tcate_count: %d'
		  % (user_count, item_count, order_count, cate_count))

	# sort history_dataframe by user_id and time
	history_df = history_df.sort_values(['user_id', 'time'])
	history_df = history_df.reset_index(drop=True)
	# reorder history dataframe's columns
	history_df = history_df[['user_id', 'order_id', 'dpt', 'dst', 'time']]

	# sort dropped dataframe by user_id and time
	dropped_df = dropped_df.sort_values(['user_id', 'time'])
	dropped_df = dropped_df.reset_index(drop=True)
	# reorder dropped dataframe's columns
	dropped_df = dropped_df[['user_id', 'order_id', 'dpt', 'dst', 'time']]
	# cate_list = [history_df[''][i] for i in range(len(history_df.index))]

	#saving preprocessed data into file.
	with open(os.path.join(aux_path, 'history.pkl'), 'wb') as f:
		pickle.dump(history_df, f)
		pickle.dump(cate_list, f)
		pickle.dump((user_count, item_count, order_count, cate_count), f)
		pickle.dump(dropped_df, f)
		print('create user history file: {}'.format(f.name))

	with open(os.path.join(aux_path, 'mapping.pkl'), 'wb') as f:
		pickle.dump((user_key, item_key, order_key), f)
		pickle.dump((user_map, item_map, order_map), f)
		print('create user item mapping: {}'.format(f.name))

	if path_user != '':
		user_df = json_to_df(path_user)

		user_df['user_id'].map(lambda x: user_map[x])
		user_df = user_df[user_df['user_id'].isin(
			history_df['user_id'].unique())]
		user_df.reset_index(drop=True)

		user_df = user_df.sort_values('user_id')
		user_df = user_df.reset_index(drop=True)

		with open(os.path.join(aux_path, 'user.pkl'), 'wb') as f:
			pickle.dump(user_df, f)
else:
	print("==========================data preprocessing setting==========================")
	print("\tsaving preprocessed aux path - %s" % aux_path)
	print("\tminimum order number - %d" % min_order)
	print("\tcutting long sequence - %s" % ("True" if long_seq else "False"))
	print("\tnonhybrid - %s" % ("True" if nonhybrid else "False"))
	print("==============================================================================")

with open(os.path.join(aux_path, 'history.pkl'), 'rb') as f:
	history_df = pickle.load(f, encoding='bytes')
	cate_list = pickle.load(f, encoding='bytes')
	user_count, item_count, order_count, cate_count = pickle.load(
		f, encoding='bytes')
	dropped_df = pickle.load(f, encoding='bytes')
with open(os.path.join(aux_path, 'mapping.pkl'), 'rb') as f:
	user_key, item_key, order_key = pickle.load(f, encoding='bytes')
	user_map, item_map, order_map = pickle.load(f, encoding='bytes')

# load and genearte travel network
travel_net_df = pd.read_csv(os.path.join('../raw_data/itny/AirNet.csv'))
travel_network = travel_net_gen(travel_net_df, item_map)

train_set = []
test_set = []

user_current_hist = {}

prev_user = 0
prev_hist = []
user_hist = {}

def gen_neg(current_airport, pos):
	"""
		generate negative item w.r.t. current positive item
	:param current_airport: last element of the prefix of city series
	:param pos: next element corresponding to the prefix of city series
	:return: negative city
	"""
	neg = pos
	# print travel_network
	while neg == pos or neg == current_airport:
		# if len(travel_network[current_airport].indices) <= 1:
		# 	# print current_airport
		# 	neg = random.randint(0, item_count-1)
		# 	continue
		# neg = random.choice(travel_network[current_airport].indices)
		neg = random.randint(0, item_count-1)
	return neg

def merge_travel_route(dpt, dst):
    """

    :param dpt: departure city
    :param dst: arrival city
    :return: travel route
    """
    i=0
    j=0
    travel_route = []
    while(i<len(dpt)):
        travel_route.append(dpt[i])
        i += 1
        if i<len(dpt) and dpt[i]!=dst[j]:
            travel_route.append(dst[j])
        j += 1
    travel_route.append(dst[-1])
    return travel_route

def train_set_clean(train_set):
	train_temp = []
	while (train_set):
		ts = train_set.pop()
		if not train_temp:
			train_temp.append(ts)
		elif ts[0][1] != train_temp[-1][0][1]:
			train_temp.append(ts)

	train_set = [(ts[0][0], ts[1], ts[2], ts[3]) for ts in reversed(train_temp)]
	return train_set

# groupby history dataframe based on user_id and order_id
for user_order_id, history in history_df.groupby(['user_id', 'order_id']):

	if user_order_id[0] != prev_user:
		prev_user = user_order_id[0]
		prev_hist = []
		if train_set:
			train_set.pop()

	pos_list 	= merge_travel_route(history['dpt'].tolist(), history['dst'].tolist())
	# user_all_hist[user_order_id[0]] = user_all_hist.get(user_order_id[0], []) + pos_list

	prev_hist = prev_hist[:-1] if prev_hist and prev_hist[-1] == pos_list[0] else prev_hist

	'''
		prev hist city tuple, each tuple item contains previous city and current city,
		use gen_neg to generate a list of negative sample w.r.t. corresponding index.
	'''
	cur_hist = prev_hist + pos_list
	city_tuple = zip(cur_hist[:-1], cur_hist[1:])
	neg_hist_i = [gen_neg(cur, pos) for cur, pos in city_tuple]

	seq_begin = 1
	# use whole travel history if nonhybrid, else use the prefix of travel history as training set
	if nonhybrid:
		seq_begin = max(len(pos_list) - 2, 1)
	for i in range(seq_begin, len(pos_list)):
		hist = pos_list[:i]

		'''
			cutting strategy for long/short sequence of history
		'''
		if long_seq:
			train_set.append((user_order_id, prev_hist+hist, pos_list[i], neg_hist_i[: len(prev_hist)+i]))
		else:
			train_set.append((user_order_id, hist, pos_list[i], neg_hist_i[len(prev_hist)+1: len(prev_hist)+i]))

		# test example
		if i==len(pos_list)-1:
			if long_seq:
				user_current_hist[user_order_id[0]] = (user_order_id[0], prev_hist+hist, pos_list[i], neg_hist_i)
			else:
			 	user_current_hist[user_order_id[0]] = (user_order_id[0], hist, pos_list[i], neg_hist_i[len(prev_hist):])
	prev_hist += pos_list
	user_hist[user_order_id[0]] = prev_hist

if nonhybrid:
	train_final_set = train_set_clean(train_set)
else:
	train_final_set = [(ts[0][0], ts[1], ts[2], ts[3]) for ts in train_set]
test_set = [test for uid, test in user_current_hist.items()]
print('training set: %d, test set: %d' % (len(train_final_set), len(test_set)))
# print '\tremove %d users with less than %d orders' % (len(remove_user, min_order))
assert len(test_set) == user_count, "user number unmatch"
assert len(user_hist.keys()) == user_count, "user number unmatch"

'''
	same process for dropped dataframe
'''
d_test_set = []
d_user_current_hist = {}

d_prev_user = 0
d_prev_hist = []
d_user_hist = {}

# groupby dropped dataframe based on user id and order_id, dropped dataframe is only used for testing purpose
for user_order_id, history in dropped_df.groupby(['user_id', 'order_id']):
	if user_order_id[0] != d_prev_user:
		d_prev_user = user_order_id[0]
		d_prev_hist = []

	pos_list 	= merge_travel_route(history['dpt'].tolist(), history['dst'].tolist())
	d_prev_hist = d_prev_hist[:-1] if d_prev_hist and d_prev_hist[-1] == pos_list[0] else d_prev_hist

	'''
		prev hist city tuple, each tuple item contains previous city and current city,
		use gen_neg to generate a list of negative sample w.r.t. corresponding index.
	'''
	cur_hist = d_prev_hist + pos_list
	city_tuple = zip(cur_hist[:-1], cur_hist[1:])
	neg_hist_i = [gen_neg(cur, pos) for cur, pos in city_tuple]
	if long_seq:
		d_user_current_hist[user_order_id[0]] = (user_order_id[0], d_prev_hist+pos_list[:-1], pos_list[-1], neg_hist_i)
	else:
		d_user_current_hist[user_order_id[0]] = (user_order_id[0], pos_list[:-1], pos_list[-1], neg_hist_i[len(prev_hist):])
	d_prev_hist += pos_list
	d_user_hist[user_order_id[0]] = d_prev_hist
d_test_set = [test for uid, test in d_user_current_hist.items()]
print('dropped test set %d' % (len(d_test_set)))

# save the training set and test set
with open(os.path.join(aux_path, 'dataset_{}.pkl'.format(('l' if long_seq else 's'))), 'wb') as f:
	pickle.dump(train_final_set, f)
	pickle.dump(test_set, f)
	pickle.dump(travel_network, f)
	pickle.dump(cate_list, f)
	pickle.dump((user_count, item_count, order_count, cate_count), f)
	pickle.dump(d_test_set, f)
	print('create dataset file: {}'.format(f.name))

with open(os.path.join(aux_path, 'user_hist_{}.pkl'.format(('l' if long_seq else 's'))), 'wb') as f:
	pickle.dump(user_hist, f)
	pickle.dump(d_user_hist, f)
	print('create user history file: {}'.format(f.name))

with open(os.path.join(aux_path, 'travel_network.pkl'), 'wb') as f:
	pickle.dump(travel_network, f)
	print('saving travel network file: {}'.format(f.name))

'''
	user-wised splitting strategy
'''

user_list = np.unique(history_df['user_id'].values)
np.random.seed(1234)
np.random.shuffle(user_list)
select_user = np.sort(user_list[:int(user_count*split_ratio)])
rest_user = np.sort(user_list[int(user_count*split_ratio)::])
print('train user: %d test user: %d'%(len(select_user), len(rest_user)))
train_history_df = history_df[history_df['user_id'].isin(select_user)]
test_history_df = history_df[history_df['user_id'].isin(rest_user)]
dropped_df = dropped_df
# print 'NAY' in item_map or 'PEK' in item_map
train_set = []
test_set = []

user_current_hist = {}

prev_user = 0
prev_hist = []
user_hist = {}

# generate train_set
for user_order_id, history in train_history_df.groupby(['user_id', 'order_id']):
	if user_order_id[0] != prev_user:
		prev_user = user_order_id[0]
		prev_hist = []
		if train_set:
			train_set.pop()
	pos_list = merge_travel_route(history['dpt'].tolist(), history['dst'].tolist())
	# user_all_hist[user_order_id[0]] = user_all_hist.get(user_order_id[0], []) + pos_list
	
	prev_hist = prev_hist[:-1] if prev_hist and prev_hist[:-1] == pos_list[0] else prev_hist
	# neg_list = [gen_neg() for i in range(len(pos_list))]
	
	'''
		prev hist city tuple, each tuple item contains previous city and current city,
		use gen_neg to generate a list of negative sample w.r.t. corresponding index.
	'''
	cur_hist = prev_hist + pos_list
	city_tuple = zip(cur_hist[:-1], cur_hist[1:])
	neg_hist_i = [gen_neg(cur, pos) for cur, pos in city_tuple]
	seq_begin = 1

	# use whole travel history if nonhybrid, else use the prefix of travel history as training set
	if nonhybrid:
		seq_begin = max(len(pos_list) - 2, 1)
	for i in range(seq_begin, len(pos_list)):
		hist = pos_list[:i]

		if long_seq:
			train_set.append((user_order_id, prev_hist+hist, pos_list[i], neg_hist_i[: len(prev_hist)+i]))
			# print("pos len {}, neg len {}".format(len(prev_hist+hist), len(prev_hist)+i-1))
		else:
			train_set.append((user_order_id, hist, pos_list[i], neg_hist_i[len(prev_hist): len(prev_hist)+i]))
	prev_hist += pos_list
	user_hist[user_order_id[0]] = prev_hist
# generate test set
if nonhybrid:
	train_final_set = train_set_clean(train_set)
else:
	train_final_set = [(ts[0][0], ts[1], ts[2], ts[3]) for ts in train_set]
for user_order_id, history in test_history_df.groupby(['user_id', 'order_id']):
	if user_order_id[0] != prev_user:
		prev_user = user_order_id[0]
		prev_hist = []
	
	pos_list = [history['dpt'].tolist()[0]] + history['dst'].tolist()

	# avoid acycle
	prev_hist = prev_hist[:-1] if prev_hist and prev_hist[-1] == pos_list[0] else prev_hist
	cur_hist = prev_hist + pos_list
	city_tuple = zip(cur_hist[:-1], cur_hist[1:])
	neg_hist_i = [gen_neg(cur, pos) for cur, pos in city_tuple]
	if long_seq:
		user_current_hist[user_order_id[0]] = (user_order_id[0], prev_hist+pos_list[:-1], pos_list[-1], neg_hist_i)
	else:
		user_current_hist[user_order_id[0]] = (user_order_id[0], pos_list[:-1], pos_list[-1], neg_hist_i[len(prev_hist):])
	prev_hist += pos_list
	user_hist[user_order_id[0]] = prev_hist
test_set = [test for uid, test in user_current_hist.items()]
print('training set: %d, test set: %d' % (len(train_final_set), len(test_set)))
# random.shuffle(train_final_set)
# random.shuffle(test_set)
# print '\tremove %d users with less than %d orders' % (len(remove_user, min_order))
assert len(test_set) == len(rest_user), "test set user number unmatch"
assert len(user_hist.keys()) ==  user_count, "user number unmatch"
with open(os.path.join(aux_path, 'dataset_u{}.pkl'.format(("l" if long_seq else "s"))), 'wb') as f:
	pickle.dump(train_final_set, f)
	pickle.dump(test_set, f)
	pickle.dump(travel_network, f)
	pickle.dump(cate_list, f)
	pickle.dump((user_count, item_count, order_count, cate_count), f)
	pickle.dump(d_test_set, f)
	print('create dataset file: {}'.format(f.name))

with open(os.path.join(aux_path, 'user_hist_u{}.pkl'.format(('l' if long_seq else 's'))), 'wb') as f:
	pickle.dump(user_hist, f)
	pickle.dump(d_user_hist, f)
	print('create user history file: {}'.format(f.name))

