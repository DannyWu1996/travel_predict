import numpy as np
import json
import pickle
import random

import gzip
# import shuffle

random.seed(1234)

def unicode_to_utf8(d):
	return dict((key.encode('UTF-8'), value) for key, value in d.iteritems())

class DataIterator:

    def __init__(
        self, data, 
        travel_network, 
        item_count, 
        cate_list, 
        batch_size=128,
        max_len=100,
        shuffle_each_epoch=False,
        type=1
        ):

        self.ori_data = data
        if shuffle_each_epoch:
            self.data = random.shuffle(self.ori_data)
        else:
            self.data = self.ori_data

        self.data = data
        self.travel_network = travel_network
        self.item_count = item_count
        self.cate_list = cate_list
        self.max_len = max_len
        self.batch_size = batch_size

        self.epoch_size = len(self.data) // self.batch_size
        # if self.epoch_size * self.batch_size < len(self.data):
        #     self.epoch_size += 1
        self.epoch_index = 0

        self.shuffle = shuffle_each_epoch

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.data = random.shuffle(self.data)
        else:
            self.data = self.ori_data

    def __next__(self):
        if self.epoch_index == self.epoch_size:
            raise StopIteration


        ts = self.data[self.epoch_index*self.batch_size: min((self.epoch_index+1)*self.batch_size, len(self.data))]

        self.epoch_index += 1
        u_batch, item_batch, cate_batch, label_batch, sl, sl_neg = [], [], [], [], [], []
        for t in ts:
            u_batch.append(t[0])
            item_batch.append(t[2])
            cate_batch.append(self.cate_list[t[2]])
            label_batch.append(1)
            sl.append(len(t[1]))
            sl_neg.append(len(t[3]))

            if type==1:
                neg_label = t[3][-1]
                u_batch.append(t[0])
                item_batch.append(neg_label)
                cate_batch.appned(self.cate_list[neg_label])
                label_batch.append(0)
                sl.append(len(t[1]))
                sl.neg_append(len(t[3]))

            # print('neg: {}, sl: {}'.format(sl_neg[-1], sl[-1]))
            assert sl_neg[-1] == sl[-1], "negative element unmatch"
        max_sl = max(sl)
        mask  = np.zeros([self.batch_size, max_sl], np.int64)
        hist_i = np.zeros([self.batch_size, max_sl], np.int64)
        hist_cate = np.zeros([self.batch_size, max_sl], np.int64)

        neg_hist_i = np.zeros([self.batch_size, max_sl], np.int64)
        neg_hist_cate = np.zeros([self.batch_size, max_sl], np.int64)

        for indx, t in enumerate(ts):
            # city_tuple = zip(t[1][:-1], t[1][1:])
            for l in range(len(t[1])):
                hist_i[indx][l] = t[1][l]
                hist_cate[indx][l] = self.cate_list[t[1][l]]
                mask[indx][l] = 1            
            for l in range(len(t[3])):
                neg_hist_i[indx][l] = t[3][l]
                neg_hist_cate[indx][l] = self.cate_list[t[3][l]]

        return self.epoch_index, (u_batch, item_batch, cate_batch, hist_i, hist_cate, mask, label_batch, sl, neg_hist_i, neg_hist_cate)


