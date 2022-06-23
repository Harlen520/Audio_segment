
from paddle.nn import Layer
import paddle
import paddle.nn.functional as F
from itertools import combinations
import numpy as np
from sklearn.metrics import roc_auc_score

def binary_auc_score(y_true, y_score, delta=1):
    """
    Binary auc score.
    y_true:[samples]
    y_score:[samples]
    """
    if len(np.unique(y_true)) != 2:
        raise ValueError("Only one class present in y_true. ROC AUC score "
                         "is not defined in that case.")
    pos_score = y_score[y_true]
    neg_score = y_score[paddle.logical_not(y_true)]
    pos_n = pos_score.shape[0]
    neg_n = neg_score.shape[0]
    mpr = pos_score.reshape((1, pos_n)) - neg_score.reshape((neg_n, 1)) # [neg_n, pos_n] or [pos_n, neg_n]
    auc_score = F.sigmoid(paddle.flatten(mpr) * delta)  # [pos_n * neg_n]
    return paddle.mean(auc_score)

class AucLoss(Layer):
    def __init__(self,
                 mutil_class="ovo",
                 delta=1,
                 reduction='mean',
                 use_softmax=True,
                 name=None):
        super(AucLoss, self).__init__()
        name2funct = {
            "binary": self.auc_binary_score, 
            "ovo": self.auc_ovo_score,
            "ovr": self.auc_ovr_score
        }
        self.loss_fnt=name2funct[mutil_class]
        self.delta=delta
        self.reduction = reduction
        self.use_softmax = use_softmax
        self.name = name

    def forward(self, input, label):
        if input.shape[0] != label.shape[0]:
            raise ValueError("input.shape[0] != label.shape[0]")
        return 1 - self.loss_fnt(input, label)
    
    def auc_ovo_score(self, input, label):
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

            ij_auc_score = binary_auc_score(i_true, input[ij_mask][:, ci], delta=self.delta)
            ji_auc_score = binary_auc_score(j_true, input[ij_mask][:, cj], delta=self.delta)
            pair_scores[ix] = (ij_auc_score + ji_auc_score) / 2
        
        return paddle.mean(pair_scores)
        

    def auc_ovr_score(self, input, label):
        if self.use_softmax:
            input = F.softmax(input)
        label_unique = paddle.unique(label)
        n_class = label_unique.shape[0]
        pair_scores = paddle.empty([n_class])
        for ix, ci in enumerate(label_unique):
            ci_mask = label == ci
            ij_auc_score = binary_auc_score(ci_mask, input[:, ci], delta=self.delta)
            pair_scores[ix] = ij_auc_score
        return paddle.mean(pair_scores)
        
    def auc_binary_score(self, input, label):
        
        if self.use_softmax:
            input = F.softmax(input)
        label_unique = paddle.unique(label)
        if len(label_unique) == 1:
            return paddle.to_tensor([0.5], place=input.place)
        c1_mask = label == label_unique[1]
        
        if len(input.shape) == len(label.shape):
            y_pros = input
        else:
            y_pros = input[:, label_unique[1]]
        auc_score = binary_auc_score(c1_mask, y_pros, delta=self.delta)
        
        return auc_score
