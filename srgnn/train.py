from __future__ import division
import numpy as np
from model import *
from utils import build_graph, Data, split_validation, remove_single_node
import pickle
import argparse
import datetime
import tensorflow as tf
from scipy.sparse import csr_matrix
import os
import math



parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=str, help='CUDA Device id for utilizing GPU', default=0)
parser.add_argument("--aux_path", type=str, help="Path to training data")
parser.add_argument("-m", "--method", default="ggnn", type=str, help="ggnn/gat/gcn")
parser.add_argument("-v", "--validation", action="store_true", help="validation")
parser.add_argument("-tb", "--train_batch_size", type=int, help="training batch size", default=32)
parser.add_argument("-pb", "--predict_batch_size", type=int, help="predict batch size", default=32)
parser.add_argument("-r", "--max_epoch", type=int, help="maximum round for training", default=50)
parser.add_argument("-k", "--top_k", type=int,	help="top k recommendations", default=5)
parser.add_argument("-lr", "--lr", default=1e-4, type=float, help='learning rate')
parser.add_argument("-hs", "--hiddenSize", type=int, default=100, help='hidden state size')
parser.add_argument("--l2", type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--nonhybrid', action='store_true', help='global preference')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
opt = parser.parse_args()

cuda = opt.cuda
train_batch_size = opt.train_batch_size
test_batch_size = opt.predict_batch_size
predict_batch_size = opt.predict_batch_size
aux_path = opt.aux_path
max_epoch = opt.max_epoch
top_k = opt.top_k
method = opt.method
lr = opt.lr
decay_rate = opt.lr_dc
decay_epoch = opt.lr_dc_step
hidden_size =  opt.hiddenSize
non_hybrid = opt.nonhybrid
l2_reg = opt.l2
os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda

print("==========================model setting==========================")
print("\ttraining aux path - %s" % aux_path)
print("\tvisible cuda device - %s" % cuda)
print("\ttraining batch size - %d" % train_batch_size)
print("\tpredict batch size - %d" % predict_batch_size)
print("\tmax epoch - %d" % max_epoch)
print("\tdecay rate %f" % decay_rate)
print("\tdecay after %d epochs" % decay_epoch)
print("\ttop %d prediction" % top_k)
print("\tmodel: %s" % method)
print("\tlearning rate: %f" % lr)
print("\thidden size: %d" % hidden_size)
print("\tnonhybrid: %s" % ("True" if non_hybrid else "False"))
print("\tl2: %f" % l2_reg)
print("==================================================================")

root_dir  = aux_path
'''
	load the data
'''
dataset_dir = os.path.join(root_dir, 'dataset_s.pkl')
with open(dataset_dir, 'rb') as f:
	train_set = pickle.load(f, encoding='bytes')
	test_set = pickle.load(f, encoding='bytes')
	travel_network = pickle.load(f, encoding='bytes')
	cate_list = pickle.load(f, encoding='bytes')
	user_count, item_count, order_count, cate_count = pickle.load(f, encoding='bytes')
n_node = item_count

batch_size = train_batch_size
train_data = remove_single_node(train_set)
test_data = remove_single_node(test_set)

train_data = Data(train_data, sub_graph=True, method=method, shuffle=True)
test_data = Data(test_data, sub_graph=True, method=method, shuffle=False)
model = GGNN(hidden_size=hidden_size, out_size=hidden_size, batch_size=batch_size, n_node=n_node,
				 lr=lr, l2=l2_reg, step=max_epoch, decay=decay_epoch * len(train_data.inputs) / batch_size, lr_dc=decay_rate,
				 nonhybrid=non_hybrid)
print(opt)
best_result = [0, 0]
best_epoch = [0, 0]
for epoch in range(max_epoch):
	print('epoch: ', epoch, '===========================================')
	slices = train_data.generate_batch(model.batch_size)
	fetches = [model.opt, model.loss_train, model.global_step]
	print('start training: ', datetime.datetime.now())
	loss_ = []
	for i, j in zip(slices, np.arange(len(slices))):
		adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)
		_, loss, _ = model.run(fetches, targets, item, adj_in, adj_out, alias,  mask)
		if math.isnan(loss):
			print(targets)
		# print(loss)
		loss_.append(loss)
	loss = np.mean(loss_)
	slices = test_data.generate_batch(model.batch_size)
	print('start predicting: ', datetime.datetime.now())
	hit, mrr, test_loss_ = [], [],[]
	predict_set = []
	correct_set = set()
	for i, j in zip(slices, np.arange(len(slices))):
		adj_in, adj_out, alias, item, mask, targets = test_data.get_slice(i)
		scores, test_loss = model.run([model.score_test, model.loss_test], targets, item, adj_in, adj_out, alias,  mask)
		test_loss_.append(test_loss)
		index = np.argsort(scores, 1)[:, -top_k:]
		predict_set.append(index)
		for score, target in zip(index, targets):
			hit.append(np.isin(target, score))
			if len(np.where(score == target)[0]) == 0:
				mrr.append(0)
			else:
				correct_set.add(target)
				mrr.append(1 / (top_k-np.where(score == target)[0][0]))
	hit = np.mean(hit)*100
	mrr = np.mean(mrr)*100
	print("current recall@%d:\t%.4f\tMMR@%d:\t%.4f\tEpoch:\t%d"%(top_k, hit, top_k, mrr, epoch))
	predict_set = np.unique(np.reshape(np.stack(predict_set, 0), [-1]))
	test_loss = np.mean(test_loss_)
	if hit >= best_result[0]:
		best_result[0] = hit
		best_epoch[0] = epoch
	if mrr >= best_result[1]:
		best_result[1] = mrr
		best_epoch[1]=epoch
	print('train_loss:\t%.4f\ttest_loss:\t%4f\tRecall@%d:\t%.4f\tMMR@%d:\t%.4f\tEpoch:\t%d,\t%d'%
		  (loss, test_loss, top_k, best_result[0], top_k, best_result[1], best_epoch[0], best_epoch[1]))
	print('predict set length:\t{}\tcorrect set:\t{}'.format(len(predict_set), len(correct_set)))