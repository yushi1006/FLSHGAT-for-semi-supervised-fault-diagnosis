# This is the implement of Feature–label similarity-aware hypergraph attention network: a semi-supervised fault diagnosis method for machinery under extremely low label rate scenarios
# written by Lei Wang
# Southwest Jiaotong University
# wang_llei@163.com
#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import warnings
import torch
from torch import nn
from torch import optim
import numpy as np
from datasets.UoO_hypergraph import get_files
from datasets.Label_mask import *
import models


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        self.datasets_train, self.datasets_test = get_files(args.data_dir, args.signal_length, args.sample_size, args.train_size, args.InputType)

        # Define the model
        self.model = getattr(models, args.model_name)(2048, args.num_class)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                    weight_decay=args.weight_decay)

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train_test(self):
        args = self.args
        best_acc = 0.0

        labels_train_mask, unmask_idx, mask_idx = Label_mask(self.datasets_train, train_size=args.train_size,
                                                             unmask_size=args.unmask_size, num_class=args.num_class)
        inputs_train = self.datasets_train.to(self.device)
        inputs_test = self.datasets_test.to(self.device)
        labels_train = inputs_train.y
        labels_test = inputs_test.y
        labels_train_mask = labels_train_mask.to(self.device)
        labels_test_mask = inputs_test.y
        labels_test_mask = F.one_hot(labels_test_mask, num_classes=args.num_class).to(torch.float)
        labels_test_mask[:] = 0
        unmask_idx = unmask_idx.to(self.device)
        mask_idx = mask_idx.to(self.device)

        for epoch in range(args.max_epoch):

            # training process
            self.model.train()
            with torch.set_grad_enabled(True):
                logits, labels_hat = self.model(inputs_train, labels_train_mask)
                loss1 = self.criterion(logits[unmask_idx], labels_train[unmask_idx])
                loss2 = self.criterion(labels_hat[unmask_idx], labels_train[unmask_idx])
                if len(mask_idx) != 0:
                    loss3 = self.criterion(logits[mask_idx], labels_hat[mask_idx].argmax(dim=1))
                else:
                    loss3 = 0
                loss = loss1 + loss2 + loss3

                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, labels_train).float().sum().item()
                trian_loss = loss
                trian_acc = correct / labels_train.size(0)
                labels_train_mask[mask_idx] = labels_hat.detach()[mask_idx]

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
                else:
                    logging.info('current lr: {}'.format(args.lr))

            # test process
            self.model.eval()
            logits, labels_hat = self.model(inputs_test, labels_test_mask)  # 输入测试集
            loss1 = self.criterion(logits, labels_test)
            loss2 = self.criterion(labels_hat, labels_test)
            loss3 = self.criterion(logits, labels_hat.argmax(dim=1))
            loss = loss1 + loss2 + loss3

            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, labels_test).float().sum().item()
            test_loss = loss
            test_acc = correct / labels_test.size(0)
            labels_test_mask = labels_hat.detach()


            logging.info('Epoch: {} train-Loss: {:.6f} train-Acc: {:.4f} \n test-Loss: {:.6f} test-Acc: {:.4f}'.format(
                epoch, trian_loss, trian_acc, test_loss, test_acc))

            # save the checkpoint for other learning
            model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
            # save the best model according to the val accuracy
            if test_acc > best_acc or epoch > args.max_epoch - 2:
                best_acc = test_acc
                logging.info("save best model epoch {}, acc {:.4f}".format(epoch, test_acc))
                torch.save(model_state_dic,
                           os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))






