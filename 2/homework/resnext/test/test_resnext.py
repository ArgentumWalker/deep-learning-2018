from torch.utils.data import DataLoader, TensorDataset

from resnext.resnext import *
from resnext.utils import Trainer
from sklearn.datasets import make_classification
import numpy as np
import torch


def test_random():
    X, Y = make_classification(96, 3 * 224 * 224, n_informative=20, n_classes=3)
    X = np.array(X).reshape((96, 3, 224, 224))

    train_dataset = TensorDataset(
        torch.tensor(X[:64]).type(torch.FloatTensor),
        torch.tensor(Y[:64]).type(torch.LongTensor)
    )
    test_dataset = TensorDataset(
        torch.tensor(X[64:]).type(torch.FloatTensor),
        torch.tensor(Y[64:]).type(torch.LongTensor)
    )

    trainer = Trainer(DataLoader(train_dataset, batch_size=8, shuffle=True),
                      DataLoader(test_dataset, batch_size=8, shuffle=False))
    trainer.enable_log("test_random")

    trainer.train(resnext50(num_classes=3), epochs=20)
