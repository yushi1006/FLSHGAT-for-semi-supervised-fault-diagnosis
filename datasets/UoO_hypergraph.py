# This is the implement of Feature–label similarity-aware hypergraph attention network: a semi-supervised fault diagnosis method for machinery under extremely low label rate scenarios
# written by Lei Wang
# Southwest Jiaotong University
# wang_llei@163.com
import os
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F
from scipy.io import loadmat
import torch
from datasets.Gen_hypergraph import *
from datasets.Label_mask import *


def get_files(root, signal_length=4096, sample_size=150, train_size=100, InputType="FD"):
    sub_dir = []
    data_train = []
    label_train = []
    data_test = []
    label_test = []

    datasetname = ["H-A-1.mat", "I-A-1.mat", "O-A-1.mat", "B-A-1.mat", "C-A-1.mat"]
    file_name = os.listdir(root)  # all fault modes
    for i in range(len(file_name)):
        sub_dir.append(os.path.join(root, file_name[i], datasetname[i]))
    sub_dir[4], sub_dir[0] = sub_dir[0], sub_dir[4]

    for i in tqdm(range(len(sub_dir))):
        if i == len(sub_dir) - 1:
            data_train0, label_train0, data_test0, label_test0 = data_load(sub_dir[i], signal_length, 300, 200,
                                                                           InputType, label=i)
        else:
            data_train0, label_train0, data_test0, label_test0 = data_load(sub_dir[i], signal_length, sample_size,
                                                                           train_size, InputType, label=i)
        data_train += data_train0
        label_train += label_train0
        data_test += data_test0
        label_test += label_test0

    graphset_train = Gen_hypergraph(data_train, label_train)
    graphset_test = Gen_hypergraph(data_test, label_test)
    return graphset_train, graphset_test


def data_load(root, signal_length, sample_size, train_size, InputType, label):
    fl = loadmat(root)['Channel_1']
    fl = fl.reshape(-1,)

    data = []
    lab = []
    start, end = 0, signal_length

    for i in range(sample_size):
        if InputType == "TD":
            x = fl[start:end]
            x = (x - x.min()) / (x.max() - x.min())  # normalization
        elif InputType == "FD":
            x = fl[start:end]
            x = (x - x.min()) / (x.max() - x.min())  # normalization
            x = np.fft.fft(x)
            x = np.abs(x) / len(x)
            x = x[range(int(x.shape[0] / 2))]
        else:
            print("The InputType is wrong!!")
        data.append(x)
        lab.append(label)
        start += signal_length
        end += signal_length

    data_train = data[:train_size]
    label_train = lab[:train_size]
    data_test = data[train_size:]
    label_test = lab[train_size:]

    return data_train, label_train, data_test, label_test


if __name__ == "__main__":
    data_dir = r"D:\dataset_IFD\UoO time-varying bearing dataset"
    datasets_train, datasets_test = get_files(data_dir)
    labels_train_mask, unmask_idx, mask_idx = Label_mask(datasets_train, train_size=100, unmask_size=1, num_class=5)
    labels_test_mask = datasets_test.y
    labels_test_mask = F.one_hot(labels_test_mask, num_classes=5).to(torch.float)
    labels_test_mask[:] = 0
    data_train = datasets_train.x
    label_train = datasets_train.y
    data_val = datasets_test.x
    label_val = datasets_test.y
    print(data_train.size())
    print(label_train)
    print(data_val.size())
    print(label_val)
    print(labels_train_mask)
    print(label_val.size())
    print(unmask_idx)
    print(mask_idx)
