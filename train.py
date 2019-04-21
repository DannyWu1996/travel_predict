import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from data_iterator import *
from model import *
# from model_aux import _get_top_k_recommendation
from model_aux import *

import argparse
from collections import defaultdict
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser()
parser.add_argument("-cu", "--cuda", type=str,
                    help='CUDA Device id for utilizing GPU', default=0)
parser.add_argument("-a", "--aux_path", type=str, help="Path to training data")
parser.add_argument("-tb", "--train_batch_size", type=int,
                    help="training batch size", default=32)
parser.add_argument("-pb", "--predict_batch_size", type=int,
                    help="predict batch size", default=32)
parser.add_argument("-r", "--max_epoch", type=int,
                    help="maximum round for training", default=50)
parser.add_argument("-k", "--top_k", type=int,
                    help="top k recommendations", default=5)
parser.add_argument("-n", "--neg_sampling", type=bool, help="whether use neg_sampling strategy", default=False)
parser.add_argument("-f", "--framework", default='gru', type=str, help='model_selection')
parser.add_argument("-p", "--pre_train", default=False, type=bool, help='use pre-trained embedding')
parser.add_argument("-lr", "--lr", default=1e-4, type=float, help='learning rate')
parser.add_argument("-alpha", "--alpha", default=0, type=float, help='alpha for regularization')

args = parser.parse_args()

'''
    initialize parameters
'''

cuda = args.cuda
train_batch_size = args.train_batch_size
test_batch_size = args.predict_batch_size
predict_batch_size = args.predict_batch_size
aux_path = args.aux_path
max_epoch = args.max_epoch
decay_epoch = max_epoch/2
top_k = args.top_k
neg_sampling = args.neg_sampling
framework = args.framework
pre_train = args.pre_train
lr = args.lr        
alpha = args.alpha
decay_rate = 0.1
attention_size = 32
hidden_units = 16
embedding_size = 32


os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

print("==========================model setting==========================")
print("\ttraining aux path - %s" % aux_path)
print("\tvisible cuda device - %s" % cuda)
print("\ttraining batch size - %d" % train_batch_size)
print("\tpredict batch size - %d" % predict_batch_size)
print("\tmax epoch - %d" % max_epoch)
print("\tdecay after %d epochs" % decay_epoch)
print("\ttop %d prediction" % top_k)
print("\tmodel: %s" % framework)
print("\tlearning rate: %f" % lr)
print("\tneg_sampling: %s" % ("True" if neg_sampling else "False"))
print("\tpre_train: %s" % ("True" if pre_train else "False"))
print("\talpha: %f" % alpha)
print("==================================================================")

'''
    root directory
'''
root_dir = '../raw_data/itny/itny_10'

'''
    loading the data
'''
dataset_dir = os.path.join(root_dir, 'dataset_l.pkl')
with open(dataset_dir, 'rb') as f:
    train_set = pickle.load(f, encoding='bytes')
    test_set = pickle.load(f, encoding='bytes')
    travel_network = pickle.load(f, encoding='bytes')
    cate_list = pickle.load(f, encoding='bytes')
    user_count, item_count, order_count, cate_count = pickle.load(f, encoding='bytes')


'''
    generate train, eval, test dataset
'''
total_size = 10000
predict_ads_num = item_count
train_batch = train_set
test_batch = test_set
pos_train = []
for train in train_batch:
    if train[-1] == 0:
        continue
    pos_train.append(train)
# no travel network specific, learn by model
city_network = csr_matrix(np.ones([item_count, item_count]))
print('training set: %d test set: %d evaluation set: %d'%(len(train_batch), len(test_batch), len(pos_train)))

'''
    Loading pre-trained  embedding
'''
# with open(os.path.join(root_dir, 'emb/sdne/emb{}.pkl'.format(embedding_size)), 'rb') as f:
#     embedding = pickle.load(f, encoding='bytes')
# embedding = embedding.todense().reshape([-1, hidden_units])
# print(embedding.shape)

'''
    batch_training
'''
tf.reset_default_graph()
gpu_options = tf.GPUOptions(allow_growth=True)
loss_list = []
eval_recall = defaultdict(list)
test_recall = defaultdict(list)
eval_precision = defaultdict(list)
test_precision = defaultdict(list)

eval_best_recall = []
test_best_recall = []
eval_best_precision = []
test_best_precision = []

top_ks = [top_k]

'''
	model training utilities
'''
def __calculate_(sess, model, data, travel_network, item_count, cate_list,
            batch_size, shuffle_each_epoch=True, type=1):
            
    score_arr = []
    
    true_labels = []
    current_citys = []
    users_set = set()
    recall_k = {}
    precision_k = {}
#     print("test sub items")
    for _, uij in DataIterator(data, travel_network, item_count, cate_list, 
                                batch_size, shuffle_each_epoch=True, type=2):
        u, i, ic, hist_i, hist_ic, mask, y, sl, neg_hist_i, neg_hist_ic = uij
        indx = [[r, c-1] for r, c in zip(range(len(sl)), sl)]
        current_citys += hist_i[tuple(zip(*indx))].tolist()
#         select = [travel_network[current_city]>=top_k for current_city in hist_i[tuple(zip(*indx))].tolist()]
        
        true_labels+=i
        users_set.update(u)
        score_, loss, aux_loss = model.calculate_multi(sess, uij, lr)
        # print(score_.shape)
        # score_, loss, accuracy, aux_loss = model.calculate(sess, uij, lr)
        score_arr.append(score_)
    score_arr = np.concatenate(score_arr, axis=0).reshape((-1, predict_ads_num))  
    assert len(true_labels) ==  score_arr.shape[0], '{}, {}'.format(len(true_labels), score_arr.shape)
    for top_k in top_ks:
        pred_labels_k = get_top_k_recommendation(score_arr, current_citys, top_k, travel_network)
        recall_k[top_k] = get_recall_k(pred_labels_k, true_labels, top_k)
        precision_k[top_k] = get_precision_k(pred_labels_k, true_labels, top_k, len(users_set))
        if type==1:
            print('test set top%d recall: %f precision: %f' % (top_k, recall_k[top_k], precision_k[top_k]))
        elif type==2:
            print('eval set top%d recall: %f precision: %f' % (top_k, recall_k[top_k], precision_k[top_k]))

    return (current_citys, true_labels, score_arr), recall_k, precision_k

def __eval_(sess, model, eval_data, travel_network, item_count, cate_list,
            eval_batch_size, shuffle_each_epoch=True):
    score_arr = []
    true_labels = []
    current_citys = []
    users_set = set()
    users_num = 10000
    recall_k = {}
    precision_k = {}
#     print("eval sub items")
    for _, uij in DataIterator(eval_data, travel_network, item_count, cate_list,
            eval_batch_size, shuffle_each_epoch=True):
        u, i, ic, hist_i, hist_ic, mask, y, sl, neg_hist_i, neg_hist_ic = uij
        true_labels+=i
        users_set.update(u)
        indx = [[r, c-1] for r, c in zip(range(len(sl)), sl)]
        current_citys += hist_i[tuple(zip(*indx))].tolist()
        score_, loss, accuracy, aux_loss = model.calculat(sess, uij)
        score_arr.append(score_)
    score_arr = np.concatenate(score_arr, axis=0).reshape((-1, item_count))  
    assert len(true_labels) ==  score_arr.shape[0], '{}, {}'.format(len(true_labels), score_arr.shape)
    for top_k in top_ks:
        pred_labels_k = get_top_k_recommendation(score_arr, current_citys, top_k, city_network)
        recall_k[top_k] = get_recall_k(pred_labels_k, true_labels, top_k)
        precision_k[top_k] = get_precision_k(pred_labels_k, true_labels, top_k, len(users_set))
        print('eval set top%d recall: %f precision: %f' % (top_k, recall_k[top_k], precision_k[top_k]))
    return (current_citys, true_labels, score_arr), recall_k, precision_k

def union_dict(dict1, dict2):
    for key, value in dict2.items():
        dict1[key].append(value)
    return dict1


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)) as sess:
    if framework == 'gru_att_gru':
        model = Model_GRU_ATT_GRU(
            user_count, item_count, cate_count,
            attention_size, hidden_units, neg_sampling,
            embedding_size, travel_network.todense(), alpha)
    elif framework == 'gru_gru_att':
        model = Model_GRU_GRU_ATT(
            user_count, item_count, cate_count,
            attention_size, hidden_units, neg_sampling,
            embedding_size, travel_network.todense(), alpha)
    elif framework == 'gru':
        model = Model_GRU(
            user_count, item_count, cate_count,
            hidden_units, neg_sampling,
            embedding_size, travel_network.todense(), alpha)
    elif framework == 'gru_gru':
        model = Model_GRU_GRU(
            user_count, item_count, cate_count,
            hidden_units, neg_sampling,
            embedding_size, travel_network.todense(), alpha)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())    
    
    if pre_train:
        model.pre_train(sess, embedding)
        print('adding travel network')

    time.clock()
    elapsed = 0
    for epoch in range(max_epoch):
        begin = time.time()
        if epoch == decay_epoch :
            lr *= decay_rate
        losses = 0

        for _, uij in DataIterator(
            train_batch, travel_network, item_count, cate_list, 
            train_batch_size, shuffle_each_epoch=True):
            loss, accuracy, _ = model.train(sess, uij, lr)
            losses += loss
        elapsed = time.time() - begin
        loss_list.append(losses)
        print('epoch {}, loss: {}, elapse: {:.2f} seconds'.format(epoch, losses, elapsed))

        # test for evaluation set
        _, recall, precision = __calculate_(
            sess, model, pos_train, travel_network, item_count, cate_list, 
            predict_batch_size, shuffle_each_epoch=True, type=2)
        
        eval_recall = union_dict(eval_recall, recall)
        eval_precision = union_dict(eval_precision, precision)
        
        # test for test set
        _, recall, precision = __calculate_(
            sess, model, test_batch, travel_network, item_count, cate_list,
            predict_batch_size, shuffle_each_epoch=True, type=1)
    
            
        test_recall = union_dict(test_recall, recall)
        test_precision = union_dict(test_precision, precision)


    # item_embedding = sess.run(model.item_emb_w)
    
#         print(np.amax(output, axis=1))
    for top_k in top_ks:
        eval_best_recall.append(max(eval_recall[top_k]))
        eval_best_precision.append(max(eval_precision[top_k]))
        print('eval set best top{} recall: {} precision: {}'
              .format(top_k, eval_best_recall[-1], eval_best_precision[-1]))
        
        test_best_recall.append(max(test_recall[top_k]))
        test_best_precision.append(max(test_precision[top_k]))
        print('test set best top{} recall: {} precision: {}'
              .format(top_k, test_best_recall[-1], test_best_precision[-1]))
    