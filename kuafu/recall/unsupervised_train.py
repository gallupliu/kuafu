#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:gallup
@file:unsupervised_train.py
@time:2022/05/13
"""

from model.simcse import simcse
from utils.data import texts_to_ids, load_data, SimCseDataGenerator
from utils.losses import simcse_loss
import tensorflow as tf

if __name__ == '__main__':
    data_path = '../../data/'
    # 1. bert config
    model_path = '/Users/gallup/data/bert/chinese_L-12_H-768_A-12'
    # model_path = '/Users/jifei/models/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12'
    checkpoint_path = '%s/bert_model.ckpt' % model_path
    config_path = '%s/bert_config.json' % model_path
    dict_path = '%s/vocab.txt' % model_path

    # 2. set hyper parameters
    max_len = 64
    pooling = 'cls'  # in ['first-last-avg', 'last-avg', 'cls', 'pooler']
    dropout_rate = 0.1
    batch_size = 64
    learning_rate = 5e-5
    epochs = 2
    output_units = 128
    activation = 'tanh'

    # 3. data generator
    train_data = load_data(data_path+'/unsup_sample.csv', delimiter='\t')

    train_generator = SimCseDataGenerator(train_data, dict_path, batch_size, max_len, text_tuple_size=1)
    # print(next(train_generator.forfit()))

    # 4. build model
    model = simcse(config_path, checkpoint_path, dropout_rate=dropout_rate, output_units=output_units,
                   output_activation=activation)

    # 5. model compile
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss=simcse_loss, optimizer=optimizer)

    # 6. model fit
    model.fit(train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=epochs)

