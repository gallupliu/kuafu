#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:gallup
@file:basemodel.py
@time:2022/05/13
"""
import numpy as np
import math
import random
import time
import tensorflow as tf
from transformers import AutoConfig, AutoTokenizer, TFAutoModel

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)


# backbone = TFAutoModel.from_pretrained(MODEL_NAME)


class UnsuperviseData(tf.keras.utils.Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_x = batch_x + batch_x
        bx = np.array([batch_x[i::self.batch_size] for i in range(self.batch_size)]).flatten().tolist()
        return self._tokenizer(bx)

    def _tokenizer(self, x):
        return tokenizer(x, max_length=50, padding=True, truncation=True, return_tensors="tf")


class SuperviseData(tf.keras.utils.Sequence):
    def __init__(self, query_set, doc_set, corpus, batch_size):
        self.querys = query_set
        self.docs = doc_set
        self.corpus = corpus
        self.batch_size = batch_size
        self.size = len(self.corpus)

    def __len__(self):
        return math.ceil(len(self.querys) / self.batch_size)

    def __getitem__(self, idx):
        batch_query = self.querys[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_doc = self.docs[idx * self.batch_size: (idx + 1) * self.batch_size]
        # naive in-batch negativate
        randix = random.randint(1, self.batch_size - 1)
        neg_doc = batch_doc[randix:] + batch_doc[:randix]
        bx = np.array([(batch_query[i], batch_doc[i], neg_doc[i]) for i in range(self.batch_size)]).flatten().tolist()
        return self._tokenizer(bx)

    def _tokenizer(self, inputs):
        return tokenizer(inputs, max_length=50, padding=True, truncation=True, return_tensors="tf")


def unsupervise_loss(y_pred, alpha=0.05):
    idxs = tf.range(y_pred.shape[0])
    y_true = idxs + 1 - idxs % 2 * 2  # [1 0 3 2 5 4]
    y_pred = tf.math.l2_normalize(y_pred, dim=1)
    similarities = tf.matmul(y_pred, y_pred, adjoint_b=True)
    similarities = similarities - tf.eye(tf.shape(y_pred)[0]) * 1e12
    similarities = similarities / alpha  # (6,6)
    print(y_true)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, similarities, from_logits=True)  # softmax (6,)
    return tf.reduce_mean(loss)


def supervise_loss(y_pred, alpha=0.05):
    row = tf.range(0, y_pred.shape[0], 3)  # 0 3
    col = tf.range(y_pred.shape[0])
    col = tf.squeeze(tf.where(col % 3 != 0), axis=1)  # 1 2 4 5
    y_true = tf.range(0, len(col), 2)  # [0 2]
    y_pred = tf.math.l2_normalize(y_pred, dim=1)
    similarities = tf.matmul(y_pred, y_pred, adjoint_b=True)

    similarities = tf.gather(similarities, row, axis=0)
    similarities = tf.gather(similarities, col, axis=1)

    similarities = similarities / alpha  # (2,4)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, similarities, from_logits=True)
    return tf.reduce_mean(loss)


class baseModel(tf.keras.Model):
    def __init__(self, MODEL_NAME, finetune=False):
        super().__init__()
        self.backbone = TFAutoModel.from_pretrained(MODEL_NAME)
        if not finetune:
            self.backbone.trainable = False
            print("bert close")
        self.drop = tf.keras.layers.Dropout(0.2)
        self.dense_layer = tf.keras.layers.Dense(128)

    def call(self, inputs, training=False):
        x = self.backbone(inputs)[1]
        # x = self.drop(x)
        x = self.dense_layer(x)
        return x


if __name__ == '__main__':
    model = baseModel(MODEL_NAME, finetune=False)
    #unsuper
    # epochs = 5
    # batch_size = 64
    #
    # t0 = time.time()
    # for i in range(epochs):
    #     ds = UnsuperviseData(doc_df["doc_content"].values.tolist(), batch_size)
    #     print(f"epoch {i}, training ")
    #     for step, batchx in enumerate(ds):
    #         with tf.GradientTape() as tape:
    #             y_pred = model(batchx, training=True)
    #             loss = unsupervise_loss(y_pred)
    #         gradients = tape.gradient(loss, model.trainable_variables)
    #         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #         if step % 50 == 0:
    #             print("Iteration step: {}; Loss: {:.3f}, Accuracy: {:.3%}, spend time: {:.3f}".format(step, loss, 0,
    #                                                                                                   time.time() - t0))

    #sup
    epochs = 5
    batch_size = 32

    t0 = time.time()
    for i in range(epochs):
        ds = SuperviseData(train_data["query_content"].values.tolist(), train_data["doc_content"].values.tolist(),
                           doc_df["doc_content"].values.tolist(), batch_size)
        print(f"epoch {i}, training ")
        for step, batchx in enumerate(ds):
            with tf.GradientTape() as tape:
                y_pred = model(batchx, training=True)
                loss = supervise_loss(y_pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if step % 50 == 0:
                print("Iteration step: {}; Loss: {:.3f}, Accuracy: {:.3%}, spend time: {:.3f}".format(step, loss, 0,
                                                                                                      time.time() - t0))
