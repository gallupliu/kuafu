#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:gallup
@file:siamese.py
@time:2022/05/17
"""

from transformers import BertPreTrainedModel
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(BertPreTrainedModel):
    def __init__(self, model_path, config):
        super().__init__(config)
        self.bert = BertModel.from_pretrained(model_path)

    def encode(self, encoded_input):
        model_output = self.bert(**encoded_input)
        input_embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return input_embedding

    def mean_pooling(self, model_output, attention_mask):
        token_embedding = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embedding.size()).float()
        # sum columns
        sum_embddings = torch.sum(token_embedding * input_mask_expanded, 1)
        # sum_mask
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embddings / sum_mask

    def forward(self, encoded_sent1, encoded_sent2):
        sent1_embedding = self.encode(encoded_sent1)
        sent2_embedding = self.encode(encoded_sent2)
        # cos_score = F.cosine_similarity(sent1_embedding, sent2_embedding, dim=1)
        return sent1_embedding, sent2_embedding

class BertForCoSentNetwork(BertPreTrainedModel):

    def __init__(self, model_path, config):
        super().__init__(config)
        self.bert = BertModel.from_pretrained(model_path)

    def encode(self, encoded_input):
        model_output = self.bert(**encoded_input)
        input_embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return input_embedding

    def mean_pooling(self, model_output, attention_mask):
        token_embedding = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embedding.size()).float()
        # sum columns
        sum_embddings = torch.sum(token_embedding * input_mask_expanded, 1)
        # sum_mask
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embddings / sum_mask

    def forward(self, encoded_sent1, encoded_sent2, label_ids=None, λ=20):
        sent1_embedding = self.encode(encoded_sent1)  # batch_size * hidden_size
        sent2_embedding = self.encode(encoded_sent2)  # batch_size * hidden_size
        # cos_score = F.cosine_similarity(sent1_embedding, sent2_embedding, dim=1)
        sent1_norm_embedding = F.normalize(sent1_embedding, p=2, dim=1, eps=1e-8)  # l2 正则化
        sent2_norm_embedding = F.normalize(sent2_embedding, p=2, dim=1, eps=1e-8)  # l2 正则化
        if label_ids is None:
            return sent1_norm_embedding, sent2_norm_embedding
        sent_cosine = torch.sum(sent1_norm_embedding * sent2_norm_embedding, dim=1) * λ  # [batch_size]
        sent_cosine_diff = sent_cosine[:, None] - sent_cosine[None, :]  # 实现 si - sj 的粗结果 (未进行条件 si < sj 的筛选)
        labels = label_ids[:, None] < label_ids[None,]  # 进行条件 si < sj 的筛选, 不满足条件的都是 False
        labels = labels.long()  # False -> 0, True -> 1
        sent_cosine_exp = sent_cosine_diff - (
                    1 - labels) * 1e12  # 满足条件 True的位置 不变， False的位置直接给一个很小的数类似于 - np.inf 的意思
        # loss function 完成形式 log(1 + ∑log(1 + e^λ(si - sj))) 这里 λ 取值为20
        sent_cosine_exp = torch.cat((torch.zeros(1).to(sent_cosine_exp.device), sent_cosine_exp.view(-1)), dim=0)
        loss = torch.logsumexp(sent_cosine_exp, dim=0)
        return sent1_norm_embedding, sent2_norm_embedding, loss


if __name__ == '__main__':
    pass
