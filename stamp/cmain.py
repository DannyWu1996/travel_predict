# coding=utf-8
from optparse import OptionParser
import tensorflow as tf
import pandas as pd
import numpy as np

from data_prepare.entity.samplepack import Samplepack

from data_prepare.itny_read_p import load_data_p


from util.Config import read_conf
from util.FileDumpLoad import dump_file, load_file
from util.Randomer import Randomer
import pickle
import os
# the data path.

root_path = "~/travel_predict"
project_name = "stamp"

# the pretreatment data path.


def load_random( item_count, pad_idx=0, edim=300, init_std=0.05):
    emb_dict = np.random.normal(0, init_std, [item_count, edim])
    return emb_dict



def load_tt_datas(config={}, reload=True):
    '''
    loda data.
    config: 获得需要加载的数据类型，放入pre_embedding.
    nload: 是否重新解析原始数据
    '''

    dataset_dir = os.path.join(config["aux_path"], config["dataset"])
    with open(dataset_dir, 'rb') as f:
        train_set = pickle.load(f, encoding='bytes')
        test_set = pickle.load(f, encoding='bytes')
        travel_network = pickle.load(f, encoding='bytes')
        cate_list = pickle.load(f, encoding='bytes')
        user_count, item_count, order_count, cate_count = pickle.load(f, encoding='bytes')
    
    print( "load the datasets.")
    train_data, test_data = load_data_p(
        train_set,
        test_set
    )

    config["n_items"] = item_count
    config["class_num"] = item_count

    emb_dict = load_random(item_count, edim=config['hidden_size'], init_std=config['emb_stddev'])
    
    config['pre_embedding'] = emb_dict
    print("-----")
    return train_data, test_data


def load_conf(model, modelconf):
    '''
    model: 需要加载的模型
    modelconf: model config文件所在的路径
    '''
    # load model config
    model_conf = read_conf(model, modelconf)
    if model_conf is None:
        raise Exception("wrong model config path.", model_conf)
    module = model_conf['module']
    obj = model_conf['object']
    params = model_conf['params']
    params = params.split("/")
    paramconf = ""
    model = params[-1]
    for line in params[:-1]:
        paramconf += line + "/"
    paramconf = paramconf[:-1]
    # load super params.
    param_conf = read_conf(model, paramconf)
    return module, obj, param_conf


def option_parse():
    '''
    parse the option.
    '''
    parser = OptionParser()

    parser.add_option(
        "--cuda",
        action="store",
        type=str,
        default='0'
    )

    parser.add_option(
        "--aux_path",
        type='str',
        default="../raw_data/itny/itny_v2_10"
    )

    parser.add_option(
        "-m",
        "--model",
        action='store',
        type='string',
        dest="model",
        default='stamp'
    )
    # parser.add_option(
    #     "-c",
    #     "--classnum",
    #     action='store',
    #     type='int',
    #     dest="classnum",
    #     default=3
    # )

    parser.add_option(
        "-a",
        "--nottrain",
        action='store_true',
        dest="not_train",
        default=False
    )
    parser.add_option(
        "-n",
        "--notsavemodel",
        action='store_true',
        dest="not_save_model",
        default=False
    )

    parser.add_option(
        "--dataset",
        type='str',
        default="dataset_l.pkl"
    )

    parser.add_option(
        "-p",
        "--modelpath",
        action='store',
        type='string',
        dest="model_path",
    )
    parser.add_option(
        "-i",
        "--inputdata",
        action='store',
        type='string',
        dest="input_data",
        default='test'
    )
    parser.add_option(
        "-e",
        "--epoch",
        action='store',
        type='int',
        dest="epoch",
        default=10
    )
    parser.add_option(
        "-b",
        "--batch_size",
        type='int',
        default=128
    )
    (option, args) = parser.parse_args()
    return option


def main(options, modelconf="config/model.conf"):
    '''
    model: 需要加载的模型
    dataset: 需要加载的数据集
    reload: 是否需要重新加载数据，yes or no
    modelconf: model config文件所在的路径
    class_num: 分类的类别
    use_term: 是否是对aspect term 进行分类
    '''
    model = options.model
    cuda = options.cuda
    aux_path = options.aux_path
    dataset = options.dataset
    reload = True
    class_num = 0
    is_train = not options.not_train
    is_save = not options.not_save_model
    model_path = options.model_path
    input_data = options.input_data
    epoch = options.epoch
    batch_size = options.batch_size

    module, obj, config = load_conf(model, modelconf)
    config['model'] = model
    print(model)
    config['aux_path'] = aux_path
    config['dataset'] = dataset
    config['nepoch'] = epoch
    config['batch_size'] = batch_size
    train_data, test_data = load_tt_datas(config, reload)
    module = __import__(module, fromlist=True)

    # setup randomer

    Randomer.set_stddev(config['stddev'])

    with tf.Graph().as_default():
        # build model
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        
        model = getattr(module, obj)(config)
        model.build_model()
        if is_save or not is_train:
            saver = tf.train.Saver(max_to_keep=30)
        else:
            saver = None
        # run
        with tf.Session(config=gpu_config) as sess:
            sess.run(tf.global_variables_initializer())
            if is_train:
                model.train(sess, train_data, test_data, saver, threshold_acc=config['base_threshold_acc'])
            else:
                if input_data is "test":
                    sent_data = test_data
                elif input_data is "train":
                    sent_data = train_data
                else:
                    sent_data = test_data
                saver.restore(sess, model_path)
                model.test(sess, sent_data)


if __name__ == '__main__':
    options = option_parse()
    main(options)
