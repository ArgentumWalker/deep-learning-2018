from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, Optimizer
import torch

from tensorboardX import SummaryWriter

ACCURACY_TAG = "Accuracy"
LOSS_TAG = "Loss"

__all__ = ["Trainer", "ScalarLogger"]


class ScalarLogger:
    def __init__(self, logdir):
        self.writer = SummaryWriter(logdir)
        self.step = 0

    def log(self, tag, value):
        self.writer.add_scalar(tag, value, self.step)

    def inc_step(self):
        self.step += 1


class Trainer:
    def __init__(self, train_dataloader:DataLoader, test_dataloader:DataLoader):
        self._train = train_dataloader
        self._test = test_dataloader
        self._train_logger = None
        self._test_logger = None

    def train(self, model:Module, optim:Optimizer=None, loss_function=None, epochs=50):
        model.cuda()
        if optim is None:
            optim = Adam(model.parameters())
        if loss_function is None:
            loss_function = CrossEntropyLoss()
        for i in range(epochs):
            (train_acc, train_loss), (test_acc, test_loss) = self.epoch(model, optim, loss_function)
            if self._train_logger is not None:
                self._train_logger.log(ACCURACY_TAG, train_acc)
                self._train_logger.log(LOSS_TAG, train_loss)
                self._train_logger.inc_step()
            if self._test_logger is not None:
                self._test_logger.log(ACCURACY_TAG, test_acc)
                self._test_logger.log(LOSS_TAG, test_loss)
                self._test_logger.inc_step()

    def epoch(self, model:Module, optim:Optimizer, loss_function=None):
        if loss_function is None:
            loss_function = CrossEntropyLoss()
        # Train
        for data, labels in self._train:
            optim.zero_grad()

            pred = model(data.cuda())

            loss = loss_function(pred, labels.cuda())
            loss.backward()
            optim.step()

        # Score
        train_score = self._score(model, self._train, loss_function)
        test_score = self._score(model, self._test, loss_function)

        return train_score, test_score

    def _score(self, model, dataset, loss_function):
        iters = 0
        avg_loss = 0.
        avg_accuracy = 0.
        for data, labels in dataset:
            iters += 1

            pred = model(data.cuda())
            avg_accuracy += (torch.max(pred, 1)[1] == labels.cuda()).sum().item() / len(data)

            loss = loss_function(pred, labels.cuda())
            avg_loss += loss.item()
        avg_loss /= iters
        avg_accuracy /= iters
        return avg_accuracy, avg_loss

    def enable_log(self, logdir):
        self._train_logger = ScalarLogger(logdir + "/train")
        self._test_logger = ScalarLogger(logdir + "/test")

    def disable_log(self):
        self._train_logger = None
        self._test_logger = None

