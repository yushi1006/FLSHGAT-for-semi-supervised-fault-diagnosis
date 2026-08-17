# This is the implement of Feature–label similarity-aware hypergraph attention network: a semi-supervised fault diagnosis method for machinery under extremely low label rate scenarios
# written by Lei Wang
# Southwest Jiaotong University
# wang_llei@163.com
import logging
import time
import warnings
import torch
import random
import numpy as np
import torch.nn.functional as F


def Label_mask(datasets_train, train_size, unmask_size, num_class):
    samples = np.array(range(train_size * (num_class+1)))
    # samples = np.array(range(train_size * num_class))
    unmask_idx, mask_idx = [], []
    labels_train_mask = datasets_train.y
    labels_train_mask = F.one_hot(labels_train_mask, num_classes=num_class).to(torch.float)

    for i in range(num_class):
        if i == num_class - 1:
            sample_c = samples[i * train_size:]
            random.seed(100)
            random.shuffle(sample_c)
            unmask_idx += torch.LongTensor(sample_c[:2 * unmask_size])  # samples with true labels
            mask_idx += torch.LongTensor(sample_c[2 * unmask_size:])  # samples without labels
        else:
            sample_c = samples[i * train_size:(i + 1) * train_size]
            random.seed(100)
            random.shuffle(sample_c)
            unmask_idx += torch.LongTensor(sample_c[:unmask_size])  # samples with true labels
            mask_idx += torch.LongTensor(sample_c[unmask_size:])  # samples without labels
        '''sample_c = samples[i * train_size:(i + 1) * train_size]
        random.seed(100)
        random.shuffle(sample_c)
        unmask_idx += torch.LongTensor(sample_c[:unmask_size])  # samples with true labels
        mask_idx += torch.LongTensor(sample_c[unmask_size:])  # samples without labels'''

    unmask_idx = torch.LongTensor(unmask_idx)
    mask_idx = torch.LongTensor(mask_idx)
    labels_train_mask[mask_idx] = 0

    return labels_train_mask, unmask_idx, mask_idx
