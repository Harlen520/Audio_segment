
from paddle.nn import Layer
import paddle
import paddle.nn.functional as F
from itertools import combinations
import numpy as np


def getQ(A, m, n):
    '''
     A must be [n, m]
    '''
    AAT = paddle.mm(A, A.T)      # [n, n]
    AAT_dia = paddle.diagonal(AAT)
    Q = (paddle.sum(AAT) - paddle.sum(AAT_dia)) / (m * n * (n-1))
    return Q

def binary_vauc_score(y_true, y_score, delta=1):
    """
    Binary auc score.
    y_true:[samples]   a set of  {True, False}
    y_score:[samples]
    """
    if len(np.unique(y_true)) != 2:
        raise ValueError("Only one class present in y_true. ROC AUC score "
                         "is not defined in that case.")
    pos_score = y_score[y_true]
    neg_score = y_score[paddle.logical_not(y_true)]
    pos_n = pos_score.shape[0]
    neg_n = neg_score.shape[0]
    
    A = pos_score.reshape((1, pos_n)) - neg_score.reshape((neg_n, 1)) # [neg_n, pos_n]
    A = A.reshape((neg_n, pos_n))
    A = F.sigmoid(A * delta) 
    auc_score = paddle.mean(A) 
    B = neg_score.reshape((1, neg_n)) - pos_score.reshape((pos_n, 1))  # [pos_n, neg_n]
    B = B.reshape((pos_n, neg_n))
    B = F.sigmoid(B * delta)
    Q0 = auc_score * (1 - auc_score) 
    Q1 = getQ(A, pos_n, neg_n)
    Q2 = getQ(B, neg_n, pos_n)
    vauc_score = (Q0 + (neg_n-1)*(Q1 - paddle.pow(auc_score, 2)) + (pos_n-1)*(Q2 - paddle.pow(1-auc_score, 2))) / ((pos_n-1)*(neg_n-1)) # [pos_n * neg_n]
    return vauc_score, auc_score

class VaucLoss(Layer):
    def __init__(self,
                 mutil_class="ovo",
                 delta=1,
                 beta=0.1,
                 reduction='mean',
                 use_softmax=True,
                 name=None):
        super(VaucLoss, self).__init__()
        name2funct = {
            "binary": self.vauc_binary_score, 
            "ovo": self.vauc_ovo_score,
            "ovr": self.vauc_ovr_score
        }
        self.loss_fnt=name2funct[mutil_class]
        self.delta=delta
        self.beta = beta
        self.reduction = reduction
        self.use_softmax = use_softmax
        self.name = name

    def forward(self, input, label):
        if input.shape[0] != label.shape[0]:
            raise ValueError("input.shape[0] != label.shape[0]")
        vauc_score = self.loss_fnt(input, label)
        
        return vauc_score
    
    def vauc_ovo_score(self, input, label):
        if self.use_softmax:
            input = F.softmax(input)
        label_unique = paddle.unique(label)
        n_class = label_unique.shape[0]
        n_pairs = n_class * (n_class - 1) // 2
        pair_scores = paddle.empty([n_pairs])
        for ix, (ci, cj) in enumerate(combinations(label_unique, 2)):
            ci_mask = label == ci
            cj_mask = label == cj
            ij_mask = paddle.logical_or(ci_mask, cj_mask)
            i_true = ci_mask[ij_mask]
            j_true = cj_mask[ij_mask]

            ij_auc_score, _ = binary_vauc_score(i_true, input[ij_mask][:, ci], delta=self.delta)
            ji_auc_score, _ = binary_vauc_score(j_true, input[ij_mask][:, cj], delta=self.delta)
            pair_scores[ix] = (ij_auc_score + ji_auc_score) / 2
        
        if self.reduction == "mean":
            return paddle.mean(pair_scores)
        elif self.reduction == "sum":
            return paddle.sum(pair_scores)
        else:
            raise ValueError("The reduction must be in ('sum', 'mean')")

    def vauc_ovr_score(self, input, label):
        if self.use_softmax:
            input = F.softmax(input)
        label_unique = paddle.unique(label)
        n_class = label_unique.shape[0]
        pair_scores = paddle.empty([n_class])
        for ix, ci in enumerate(label_unique):
            ci_mask = label == ci
            ij_auc_score, _ = binary_vauc_score(ci_mask, input[:, ci], delta=self.delta)
            pair_scores[ix] = ij_auc_score

        if self.reduction == "mean":
            return paddle.mean(pair_scores)
        elif self.reduction == "sum":
            return paddle.sum(pair_scores)
        else:
            raise ValueError("The reduction must be in ('sum', 'mean')")
        

    def vauc_binary_score(self, input, label):
        
        if self.use_softmax:
            input = F.softmax(input)
        label_unique = paddle.unique(label)
        c1_mask = label == label_unique[-1]
        if len(input.shape) == len(label.shape):
            y_pros = input
        else:
            y_pros = input[:, label_unique[1]]
        vauc_score, auc_score = binary_vauc_score(c1_mask, y_pros, delta=self.delta)
        
        return vauc_score
