# This is the implement of Feature–label similarity-aware hypergraph attention network: a semi-supervised fault diagnosis method for machinery under extremely low label rate scenarios
# written by Lei Wang
# Southwest Jiaotong University
# wang_llei@163.com
#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_hypergraph_utils import train_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='FLSHGAT', help='the name of the model')
    parser.add_argument('--result_dir', type=str, default='./results/', help='the directory of the result')
    parser.add_argument('--data_file', type=str, default='UoO01', help='the file of the data')
    parser.add_argument('--data_dir', type=str, default=r"D:\dataset_IFD\UoO time-varying bearing dataset", help='the directory of the data')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/FLSHGAT/UoO01', help='the directory to save the model')

    # core parameters
    parser.add_argument('--num_class', type=int, default=5, help='number of fault types')
    parser.add_argument('--sample_size', type=int, default=150, help='the number of samples for each fault type')
    parser.add_argument('--train_size', type=int, default=100, help='the number of train samples for each fault type')
    parser.add_argument('--unmask_size', type=int, default=1, help='the number of train samples for each fault type')
    parser.add_argument('--InputType', choices=['TD', 'FD', 'other'], type=str, default='FD',
                        help='the input type decides the length of input')
    parser.add_argument('--signal_length', type=int, default=4096, help='points of a sample')

    # optimization information
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='the weight decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step',
                        help='the learning rate schedule')
    parser.add_argument('--steps', type=str, default='100, 150', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=200, help='max number of epoch')
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train_test()


