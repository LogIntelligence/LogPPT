#-*- coding:utf-8
# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html?highlight=crf

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def batch_log_sum_exp(vec):
    max_score = torch.max(vec, dim=1).values
    max_score_broadcast = max_score.view(-1, 1)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))

class CRFLayer(nn.Module):
    def __init__(self, tag_size, required_grad = True) -> None:
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(tag_size + 2, tag_size + 2, dtype=float), requires_grad=required_grad)
        self.start_tag_id = tag_size 
        self.end_tag_id = tag_size + 1
        self.trans.data[self.start_tag_id, :] = -10000.0
        self.trans.data[:, self.end_tag_id] = -10000.0
        self.tag_size = tag_size
        self.real_tag_size = self.tag_size + 2

    def log_exp_score(self, X : torch.Tensor, y : torch.Tensor):
        '''
        X : [L, tag_size], lstm_linear的输出
        y : [L]
        '''
        score = torch.zeros(1).to(X.device)
        for i, emit in enumerate(X):
            score += self.trans[y[i], y[i-1] if i > 0 else self.start_tag_id] + emit[y[i]]
        score += self.trans[self.end_tag_id, y[-1]]
        return score

    def log_exp_sum_score(self, X : torch.Tensor):
        '''
        X : [batch_size, L, tag_size], lstm_linear的输出
        '''
        
        for i in range(X.shape[0]):

            sum_score = torch.full((1, self.real_tag_size), -10000.0).to(X.device)
            sum_score[0][self.start_tag_id] = 0
            for emit in X:
                score = sum_score + self.trans + emit.unsqueeze(0).repeat(self.real_tag_size, 1).transpose(0, 1)
                sum_score = batch_log_sum_exp(score).view(1, -1)
            sum_score += self.trans[self.end_tag_id]
        return batch_log_sum_exp(sum_score)[0]

    def nll_loss(self, X : torch.Tensor, y : torch.Tensor):
        return self.log_exp_sum_score(X) - self.log_exp_score(X, y)

    def decode(self, X : torch.Tensor) -> list[int]:
        with torch.no_grad():
            stack = []
            sum_score = torch.full((1, self.real_tag_size), -10000.0).to(X.device)
            sum_score[0][self.start_tag_id] = 0
            for emit in X:
                score = sum_score + self.trans
                best_tags = torch.argmax(score, dim=1)
                sum_score = score.gather(1, best_tags.view(-1, 1)).view(1, -1) + emit
                stack.append(best_tags)
            sum_score += self.trans[self.end_tag_id]
            best_tag_id = torch.argmax(sum_score, dim=1).item()
            path = [best_tag_id]
            for node in reversed(stack):
                best_tag_id = node[best_tag_id].item()
                path.append(best_tag_id)
            start = path.pop()
            assert start == self.start_tag_id
            path.reverse()
            return path

    def init_transitions(self, trans : torch.Tensor, require_grad=True):
        assert trans.shape[0] == self.real_tag_size and trans.shape[1] == self.real_tag_size
        self.trans = nn.Parameter(trans, requires_grad=require_grad)
        self.trans.data[self.start_tag_id, :] = -10000.0
        self.trans.data[:, self.end_tag_id] = -10000.0
    