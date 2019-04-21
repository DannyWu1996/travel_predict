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
Create on Jan 7, 2019
@author: danny.W
'''
random.seed(1234)

'''
	Transform json format file into panda's Dataframe structure.
	Parameter
	file_path: str
		Path to json file
'''


def json_to_df(file_path):
	assert os.path.exists(file_path), "file unknown!!!"
	with open(file_path, 'r') as fin:
		dic = {}
		indx = 0
		for line in fin:
			# print eval(line)
			dic[indx] = eval(line)
			indx += 1

		df = pd.DataFrame.from_dict(dic, orient='index')
		df.reset_index(drop=True)
		return df


'''
	saving pandas' dataframe into pickle file format
	parameters:
	file_path: str
		directory of saving file
	df_object: Pandas' Dataframe
		pandas' dataframe object
'''


def save_df(file_path, df_object):
	with open(file_path, 'wb') as f:
		pickle.dump(df_object, f)
	print("\tsaving dataframe to path - %s" % file_path)


'''
	build a mapping from str to int
	parameter:
	df: Pandas.Dataframe
		pandas' dataframe structure
	col_name: str
		column to call the mapping
'''


def build_map(df, col_name):
	# print df.columns.values
	if col_name == 'item':
		city_list = np.unique(np.concatenate(
			(df['dpt'].unique(), df['dst'].unique()), 0)).tolist()
		# print city_list
		key = sorted(city_list)
		m = dict(zip(key, range(len(key))))
		df['dpt'] = df['dpt'].map(lambda x: m[x])
		df['dst'] = df['dst'].map(lambda x: m[x])
	else:
		key = sorted(df[col_name].unique().tolist())
		m = dict(zip(key, range(len(key))))
		df[col_name] = df[col_name].map(lambda x: m[x])
	return m, key


'''
	split raw data from travel circle into single travel records
	Parameters:
	file_path: str
		Path to given file
'''


def do_split_data(file_path, saved_file_path, city_list):
	assert os.path.exists(file_path), "file unknown!!!"
	travel_history = []

	def remove_abroad(travel_series, valid_set):
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
	remove_records = []
	remove_user_nums = 0
	index = 0
	history_df = history_df.sort_values('user_id')
	for user_id, hist in history_df.groupby('user_id'):
		if len(hist) < min_order:
			remove_user_nums += 1
			remove_records += [i for i in range(index, index+len(hist))]
		index += len(hist)

	print("\tremove user nums: %d" % remove_user_nums)
	history_df = history_df.drop(history_df.index[remove_records])
	return history_df


'''
	return travel network with item_id
'''


def travel_net_gen(travel_net_df, item_map):

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


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--preprocess", type=bool,
					help="choose whether to process the data", default=None)
parser.add_argument("-s", "--do_split", type=str,
					help="Split raw json file to specific format. Format, uid, dpt, dst, order_id", default=None)
parser.add_argument("-r", "--raw_history_data_path", type=str,
					help="Path to raw history data. Json format, with column user_id, dpt, dst, time, order_id")
parser.add_argument("-m", "--min_order", type=int,
					help="Users who have less than \"min_order\" orders should be removed", default=0)
parser.add_argument("-a", "--aux_path", type=str,
					help="Path to processed data, including user2id, city2id mappings")
parser.add_argument("-u", "--raw_user_data_path", type=str,
					help="Path to raw user data, Json format, with columns user_id, age, gender, travel_miles", default='')
parser.add_argument("-l", "--long_seq", type=bool, help='cutting strategy for long or short sequence', default=None)
parser.add_argument("-sr", "--split_ratio", type=float, help="user_based split ratio for test and training dataset", default=0.8)
args = parser.parse_args()

do_split = args.do_split
do_preprocess = args.preprocess
path_history = args.raw_history_data_path
min_order = args.min_order
aux_path = args.aux_path
path_user = args.raw_user_data_path
long_seq = args.long_seq
split_ratio = args.split_ratio

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
	print("==============================================================================")

	with open('../raw_data/itny/city_air_map.pkl', 'rb') as f:
		city_list, air_list, air_city_map = pickle.load(f, encoding='bytes')
	if do_split is not None:
		# print do_split
		do_split_data(do_split, path_history, city_list)

	if not os.path.isdir(aux_path):
		os.mkdir(aux_path)

	history_df = json_to_df(path_history)
	history_df = remove_user_record(history_df, min_order)

	#### build the mapping for each dataframe column
	user_map, user_key = build_map(history_df, 'user_id')
	item_map, item_key = build_map(history_df, 'item')
	order_map, order_key = build_map(history_df, 'order_id')

	#     cate_list = [i for i in range(item_count)]
#     cate_list = np.array(cate_list, dtype=np.int32)

	#generate random category for each item
	cate_list = np.arange(len(item_map))

	user_count, item_count, order_count, cate_count =\
		len(user_map), len(item_map), len(order_map), len(cate_list)
	print('user_count: %d\titem_count: %d\torder_count: %d\tcate_count: %d'
		  % (user_count, item_count, order_count, cate_count))

	history_df = history_df.sort_values(['user_id', 'time'])
	history_df = history_df.reset_index(drop=True)
	history_df = history_df[['user_id', 'order_id', 'dpt', 'dst', 'time']]
	# cate_list = [history_df[''][i] for i in range(len(history_df.index))]

	#saving preprocessed data into file.
	with open(os.path.join(aux_path, 'history.pkl'), 'wb') as f:
		pickle.dump(history_df, f)
		pickle.dump(cate_list, f)
		pickle.dump((user_count, item_count, order_count, cate_count), f)
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
	print("==============================================================================")

with open(os.path.join(aux_path, 'history.pkl'), 'rb') as f:
	history_df = pickle.load(f, encoding='bytes')
	cate_list = pickle.load(f, encoding='bytes')
	user_count, item_count, order_count, cate_count = pickle.load(
		f, encoding='bytes')
with open(os.path.join(aux_path, 'mapping.pkl'), 'rb') as f:
	user_key, item_key, order_key = pickle.load(f, encoding='bytes')
	user_map, item_map, order_map = pickle.load(f, encoding='bytes')

travel_net_df = pd.read_csv(os.path.join('../raw_data/itny/AirNet.csv'))
travel_network = travel_net_gen(travel_net_df, item_map)

train_set = []
test_set = []

user_current_hist = {}

prev_user = 0
prev_hist = []
user_hist = {}

'''
	generate negative item w.r.t. current positive item
'''
def gen_neg(current_airport, pos):
	neg = pos
	# print travel_network
	while neg == pos:
		if len(travel_network[current_airport].indices) <= 1:
			# print current_airport
			neg = random.randint(0, item_count-1)
			continue
		neg = random.choice(travel_network[current_airport].indices)
	return neg


for user_order_id, history in history_df.groupby(['user_id', 'order_id']):
	if user_order_id[0] != prev_user:
		prev_user = user_order_id[0]
		prev_hist = []
	pos_list = [history['dpt'].tolist()[0]] + history['dst'].tolist()
	# user_all_hist[user_order_id[0]] = user_all_hist.get(user_order_id[0], []) + pos_list

	prev_hist = prev_hist[:-1] if prev_hist and prev_hist[:-1] == pos_list[0] else prev_hist

	'''
		prev hist city tuple, each tuple item contains previous city and current city,
		use gen_neg to generate a list of negative sample w.r.t. corresponding index.
	'''
	cur_hist = prev_hist + pos_list
	city_tuple = zip(cur_hist[:-1], cur_hist[1:])
	neg_hist_i = [gen_neg(cur, pos) for cur, pos in city_tuple]
			

	for i in range(1, len(pos_list)):
		hist = pos_list[:i]
		if i != len(pos_list) - 1:
			'''
				cutting strategy for long/short sequence of history
			'''
			if long_seq:
				train_set.append((user_order_id[0], prev_hist+hist, pos_list[i], neg_hist_i[: len(prev_hist)+i-1]))
			else:
				train_set.append((user_order_id[0], hist, pos_list[i], neg_hist_i[len(prev_hist)+1: len(prev_hist)+i-1]))
		else:
			user_current_hist[user_order_id[0]] = (user_order_id[0], prev_hist+hist, pos_list[i], neg_hist_i)
			# user_current_hist[user_order_id[0]] = (user_order_id[0], hist, pos_list[i], neg_hist_i[len(prev_hist):])
	prev_hist += pos_list
	user_hist[user_order_id[0]] = prev_hist
test_set = [test for uid, test in user_current_hist.items()]
print('training set: %d, test set: %d' % (len(train_set), len(test_set)))
# print '\tremove %d users with less than %d orders' % (len(remove_user, min_order))
assert len(test_set) == user_count, "user number unmatch"
assert len(user_hist.keys()) == user_count, "user number unmatch"
with open(os.path.join(aux_path, 'dataset_{}.pkl'.format(('l' if long_seq else 's'))), 'wb') as f:
	pickle.dump(train_set, f)
	pickle.dump(test_set, f)
	pickle.dump(travel_network, f)
	pickle.dump(cate_list, f)
	pickle.dump((user_count, item_count, order_count, cate_count), f)
	print('create dataset file: {}'.format(f.name))

with open(os.path.join(aux_path, 'user_hist.pkl'), 'wb') as f:
	pickle.dump(user_hist, f)
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
	pos_list = [history['dpt'].tolist()[0]] + history['dst'].tolist()
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
	for i in range(1, len(pos_list)):
		hist = pos_list[:i]
		if i != len(pos_list) - 1:
			if long_seq:
				train_set.append((user_order_id[0], prev_hist+hist, pos_list[i], neg_hist_i[: len(prev_hist)+i]))
				# print("pos len {}, neg len {}".format(len(prev_hist+hist), len(prev_hist)+i-1))
			else:
				train_set.append((user_order_id[0], hist, pos_list[i], neg_hist_i[len(prev_hist): len(prev_hist)+i]))
	prev_hist += pos_list
	user_hist[user_order_id[0]] = prev_hist
# generate test set
for user_order_id, history in test_history_df.groupby(['user_id', 'order_id']):
	if user_order_id[0] != prev_user:
		prev_user = user_order_id[0]
		prev_hist = []
	
	pos_list = [history['dpt'].tolist()[0]] + history['dst'].tolist()
	# user_all_hist[user_order_id[0]] = user_all_hist.get(user_order_id[0], []) + pos_list
	
	# avoid acycle
	prev_hist = prev_hist[:-1] if prev_hist and prev_hist[:-1] == pos_list[0] else prev_hist
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
print('training set: %d, test set: %d' % (len(train_set), len(test_set)))
random.shuffle(train_set)
random.shuffle(test_set)
# print '\tremove %d users with less than %d orders' % (len(remove_user, min_order))
assert len(test_set) == len(rest_user), "test set user number unmatch"
assert len(user_hist.keys()) ==  user_count, "user number unmatch"
with open(os.path.join(aux_path, 'dataset_u{}.pkl'.format(("l" if long_seq else "s"))), 'wb') as f:
	pickle.dump(train_set, f)
	pickle.dump(test_set, f)
	pickle.dump(travel_network, f)
	pickle.dump(cate_list, f)
	pickle.dump((user_count, item_count, order_count, cate_count), f)
	print('create dataset file: {}'.format(f.name))

with open(os.path.join(aux_path, 'user_hist_u{}.pkl'.format(('l' if long_seq else 's'))), 'wb') as f:
	pickle.dump(user_hist, f)
	print('create user history file: {}'.format(f.name))
