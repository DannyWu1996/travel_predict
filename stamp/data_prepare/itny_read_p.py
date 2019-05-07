import pandas as pd
import numpy as np
from data_prepare.entity.sample import Sample
from data_prepare.entity.samplepack import Samplepack


def load_data_p(train_set, test_set):
    '''
        train_set: training  data
        test_set: test_data
    '''

    # load the data
    train_data = _load_data(train_set)
    test_data = _load_data(test_set)

    return train_data, test_data

def _load_data(data):

    samplepack = Samplepack()
    samples = []
    now_id = 0
    print("I am reading")
    
    for id, ts in enumerate(data):
        sample = Sample()
        uid = ts[0]
        in_dixes = ts[1]
        out_dixes = ts[2]
        sample.id = id
        sample.uid = uid
        sample.click_items = ts[2]
        sample.in_idxes = ts[1]
        sample.neg_click = ts[3]
        samples.append(sample)

    samplepack.samples = samples
    samplepack.init_id2sample()
    return samplepack
