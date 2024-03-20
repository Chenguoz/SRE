import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from prior_template import MetaTemplate


class PriorNet(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support, repeat_num=1):
        super(PriorNet, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.repeat_num = repeat_num
        self.ls = params.ls
        self.lu = params.lu

    def set_forward(self, x, is_feature=False, return_map=True):
        if x.shape[2] == 3:
            z_support, z_query, context_prior_map = self.parse_feature(x, is_feature)
        else:
            for repeat in range(int(x.shape[2] / 3)):
                if repeat == 0:
                    z_support, z_query, context_prior_map = self.parse_feature(x[:, :, repeat * 3:repeat * 3 + 3, :, :],
                                                                               is_feature)
                else:
                    z_support_temp, z_query_temp, context_prior_map_temp = self.parse_feature(
                        x[:, :, repeat * 3:repeat * 3 + 3, :, :],
                        is_feature)
                    z_support += z_support_temp
                    z_query += z_query_temp
                    if context_prior_map_temp is not None:
                        context_prior_map += context_prior_map_temp
            z_support /= int(x.shape[2] / 3)
            z_query /= int(x.shape[2] / 3)
            if context_prior_map is not None:
                context_prior_map /= int(x.shape[2] / 3)

        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores = self.euclidean_dist(z_query, z_proto)
        if return_map:
            return scores, context_prior_map
        else:
            return scores

    def set_forward_loss(self, x):

        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)

        scores, context_prior_map = self.set_forward(x)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)

        return float(top1_correct), len(y_label), (
        self.ls * self.loss_fn(scores, y_query), self.lu * self.affinity_loss(
            context_prior_map)), scores

    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        score = -torch.pow(x - y, 2).sum(2)
        return score

    def affinity_loss(self, context_prior_map):
        one_hot_labels = F.one_hot(torch.from_numpy(np.repeat(range(self.n_way), self.n_support + self.n_query)),
                                   self.n_way)
        ideal_affinity_matrix = torch.matmul(one_hot_labels,
                                             one_hot_labels.transpose(0, 1)).cuda().float()

        BCE_LOSS = nn.BCELoss()
        bce = BCE_LOSS(context_prior_map, ideal_affinity_matrix)

        return bce
