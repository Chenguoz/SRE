import math
from sqlite3 import paramstyle
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
from tqdm import tqdm


class MetaTemplate(nn.Module):
    def __init__(self, params, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = params.n_query  # (change depends on input)
        self.feature = model_func()
        self.change_way = change_way  # some methods allow different_way classification during training and test
        self.params = params

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    @abstractmethod
    def feature_forward(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        x = Variable(x.cuda())

        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])

            (out, out_rot, context_prior_map), z_all = self.feature.forward(x, return_map=True)

            # x = self.feature.forward(x)
            # z_all, context_prior_map = self.feature_forward(x)

            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query, context_prior_map

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer):
        # print_freq = 200
        avg_euclidean_dist_loss = 0
        avg_affinity_dist_loss = 0
        avg_loss = 0
        acc_all = []
        iter_num = len(train_loader)
        if epoch == 0 or epoch == 51:
            for i, (x, _) in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
                optimizer.zero_grad()
                correct_this, count_this, loss, _ = self.set_forward_loss(x)
                acc_all.append(correct_this / count_this * 100)
                if loss[1] == 0:
                    euclidean_dist_loss = loss[0]
                    avg_euclidean_dist_loss = avg_euclidean_dist_loss + euclidean_dist_loss.item()
                    loss = euclidean_dist_loss
                else:
                    euclidean_dist_loss = loss[0]
                    affinity_dist_loss = loss[1]
                    avg_euclidean_dist_loss = avg_euclidean_dist_loss + euclidean_dist_loss.item()
                    avg_affinity_dist_loss = avg_affinity_dist_loss + affinity_dist_loss.item()
                    loss = euclidean_dist_loss + affinity_dist_loss

                loss.backward()
                optimizer.step()

                avg_loss = avg_loss + loss.item()
        else:
            for i, (x, _) in enumerate(train_loader, 0):
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
                optimizer.zero_grad()
                correct_this, count_this, loss, _ = self.set_forward_loss(x)
                acc_all.append(correct_this / count_this * 100)
                if loss[1] == 0:
                    euclidean_dist_loss = loss[0]
                    avg_euclidean_dist_loss = avg_euclidean_dist_loss + euclidean_dist_loss.item()
                    loss = euclidean_dist_loss
                else:
                    euclidean_dist_loss = loss[0]
                    affinity_dist_loss = loss[1]
                    avg_euclidean_dist_loss = avg_euclidean_dist_loss + euclidean_dist_loss.item()
                    avg_affinity_dist_loss = avg_affinity_dist_loss + affinity_dist_loss.item()
                    loss = euclidean_dist_loss + affinity_dist_loss
                loss.backward()
                optimizer.step()
                avg_loss = avg_loss + loss.item()
            # if i % print_freq == 0:
            #     print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
            #                                                             avg_loss / float(i + 1)))
        # print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
        #                                                         avg_loss / float(i + 1)))
        print('Epoch {:d} | Batch {:d}/{:d} | Euclidean Loss {:f} | Affinity_loss {:f}'.format(epoch, i,
                                                                                               len(train_loader),
                                                                                               avg_euclidean_dist_loss / float(
                                                                                                   i + 1),
                                                                                               avg_affinity_dist_loss / float(
                                                                                                   i + 1)))
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        return avg_loss / iter_num, acc_mean

    def test_loop(self, test_loader, record=None, tqdm_bar=False):
        acc_all = []
        avg_loss = 0
        iter_num = len(test_loader)
        with torch.no_grad():
            if tqdm_bar:
                for i, (x, _) in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
                    # for i, (x, _) in enumerate(test_loader, 0):
                    self.n_query = x.size(1) - self.n_support
                    if self.change_way:
                        self.n_way = x.size(0)
                    correct_this, count_this, loss, _ = self.set_forward_loss(x)
                    if loss[1] == 0:
                        euclidean_dist_loss = loss[0]
                        loss = euclidean_dist_loss
                    else:
                        euclidean_dist_loss = loss[0]
                        affinity_dist_loss = loss[1]
                        loss = euclidean_dist_loss + affinity_dist_loss

                    acc_all.append(correct_this / count_this * 100)
                    avg_loss = avg_loss + loss.item()
            else:
                for i, (x, _) in enumerate(test_loader, 0):
                    self.n_query = x.size(1) - self.n_support
                    if self.change_way:
                        self.n_way = x.size(0)
                    correct_this, count_this, loss, _ = self.set_forward_loss(x)
                    if loss[1] == 0:
                        euclidean_dist_loss = loss[0]
                        loss = euclidean_dist_loss
                    else:
                        euclidean_dist_loss = loss[0]
                        affinity_dist_loss = loss[1]
                        loss = euclidean_dist_loss + affinity_dist_loss
                    acc_all.append(correct_this / count_this * 100)
                    avg_loss = avg_loss + loss.item()
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return avg_loss / iter_num, acc_mean
