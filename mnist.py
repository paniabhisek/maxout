#/usr/bin/python3
# -*- coding: utf-8 -*-

# External library modules
import torch
import torchvision
import numpy as np

# Local library modules
from model import MaxoutMNIST
from logs import get_logger
from utils import *
from timer import total

class BenchMarkMNIST:
    def __init__(self):
        """
        Initialize dataset benchmark objects

        Initialize training and test data.
        Create the model and optimizer
        """
        self.loss = torch.nn.CrossEntropyLoss()

        self.trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                   download=True)
        self.testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                  download=True)

        self.net = MaxoutMNIST().to(device)

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.005,
                                         momentum=0.9)

        self.logger = get_logger()
        self.logger.info(device)
        self.LOGGING_MOD = 100

    def train(self, train_size, batch_step, epochs):
        """
        Train on first :py:obj:`train_size` mnist train datasets.

        Training and evaluation of MNIST using multilayer
        perceptron maxout layers

        Algo ::

            for each epoch
                for each batch
                    get batch data

                    1. forward pass the data to network
                    2. compute loss and propagate the gradient
                    3. optimize by updating the weights
                    4. Calculate accuracy

                    track total time while doing 1, 2, 3

        :param train_size: number of examples in training set
        :type train_size: :py:obj:`int`
        :param batch_step: batch size
        :type batch_step: :py:obj:`int`
        :param epochs: number of epochs
        :type epochs: :py:obj:`int`
        """
        train_data = self.trainset.train_data.to(device)
        train_labels = self.trainset.train_labels.to(device)
        for epoch in range(epochs):
            running_loss, training_loss = 0, 0
            running_time, elapsed = 0, 0
            training_acc, acc, _acc = 0, 0, 0
            examples = 0
            for batch_num, batch_i in enumerate(range(0, train_size, batch_step)):
                # get input data for current batch
                train_batch = train_data[batch_i:min(batch_i+batch_step, train_size)]
                label_batch = train_labels[batch_i:min(batch_i+batch_step, train_size)]
                examples += train_batch.size()[0]

                # convert 28 x 28 to 784 images
                train_batch = train_batch.view((train_batch.size()[0], -1)).float()

                self.optimizer.zero_grad()

                # forward + backward + optimize
                elapsed, outputs = total(self.net, train_batch, _reps=1)
                running_time += elapsed
                elapsed, loss = total(self.loss, outputs, label_batch, _reps=1)
                running_time += elapsed
                elapsed, _ = total(loss.backward, _reps=1) # propagate the gradient
                running_time += elapsed
                elapsed, _ = total(self.optimizer.step, _reps=1) # update the weights
                running_time += elapsed

                # prepare for accuracy
                _acc = num_corrects(outputs, label_batch)
                acc += _acc
                training_acc += _acc

                # loss
                running_loss += loss.item()
                training_loss += loss.item()
                if batch_i != 0 and batch_i % self.LOGGING_MOD == 0:
                    self.logger.info('Training Epoch: %d | Time: %.4fs Avg time: %.4fs '
                                     'Batch: %d Accuracy: %.2f Loss: %.4f',
                                     epoch, running_time, running_time / batch_num,
                                     batch_i, acc * 100. / examples,
                                     running_loss / batch_num)
                    # reinitialize variables
                    running_time = 0
                    acc = 0
                    examples = 0
                    running_loss = 0
            self.logger.info('Training Epoch: %d | Training Accuracy: %.4f Training Loss: %.4f',
                             epoch, training_acc / train_size,
                             training_loss / (train_size // batch_step + 1))

    def validate(self, batch_step, val_size=10000):
        """
        Evaluate on last :py:obj:`val_size` mnist training(validation)
        datasets.

        :param val_size: number of validation examples
        :type val_size: :py:obj:`int`
        :param batch_step: batch size
        :type batch_step: :py:obj:`int`
        """
        train_data = self.trainset.train_data.to(device)
        train_labels = self.trainset.train_labels.to(device)
        running_loss = 0
        running_time, elapsed = 0, 0
        acc = 0
        # start from 50000 training images
        for batch_i in range(50000, 50000 + val_size, batch_step):
            # get input data for current batch
            val_batch = train_data[batch_i:min(batch_i+batch_step, 50000+val_size)]
            label_batch = train_labels[batch_i:min(batch_i+batch_step, 50000+val_size)]

            val_batch = val_batch.view((val_batch.size()[0], -1)).float()

            self.optimizer.zero_grad()

            # forward
            elapsed, outputs = total(self.net, val_batch, is_train=False, _reps=1)
            running_time += elapsed
            elapsed, loss = total(self.loss, outputs, label_batch, _reps=1)
            running_time += elapsed

            _acc = num_corrects(outputs, label_batch)
            acc += _acc

            _loss = loss.item()
            running_loss += _loss
            if (batch_i - 50000) != 0 and batch_i % self.LOGGING_MOD == 0:
                self.logger.info('Batch: %d | Accuracy: %.4f Loss: %.4f',
                                 batch_i, _acc / val_batch.size()[0], _loss)
        running_loss /= val_size // batch_step + 1
        acc /= val_size
        self.logger.info('Validation | Time: %.4fs Accuracy: %.4f Loss: %.4f',
                         running_time, acc, running_loss)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', help='Train the model with maxout layer')
    parser.add_argument('--valid', help='Validate the model with maxout layer')
    parser.add_argument('--train_cont', help='Continue training the model with whole training data')
    args = parser.parse_args()

    benchmark = BenchMarkMNIST()
    if args.train == 'true':
        benchmark.train(50000, 64, 5)
        torch.save(benchmark.net.state_dict(), './MaxoutMNIST.pth')
    if args.valid == 'true':
        benchmark.net.load_state_dict(torch.load('./MaxoutMNIST.pth'))
        benchmark.validate(64)
    if args.train_cont == 'true':
        benchmark.net.load_state_dict(torch.load('./MaxoutMNIST.pth'))
        benchmark.train(60000, 64, 5)
        torch.save(benchmark.net.state_dict(), './MaxoutMNIST.pth')