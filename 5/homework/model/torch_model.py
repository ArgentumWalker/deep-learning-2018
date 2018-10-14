import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from itertools import product
from random import shuffle, randint
from math import ceil

from utils.utils import process_data


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.center_embedding = nn.Embedding(vocab_size, embed_size)
        self.target_embedding = nn.Embedding(vocab_size, embed_size)

    """
    center_words shape is [batch_size x 1]
    target_words shape is [batch_size x 1]
    """

    def forward(self, center_words, target_words):
        xc = self.center_embedding(center_words).unsqueeze(1)
        xt = self.target_embedding(target_words).unsqueeze(2)
        return torch.bmm(xc, xt).squeeze()

    def get_embeddings(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        return self.center_embedding(torch.tensor([w for w in range(self.vocab_size)], dtype=torch.long).to(device))\
            .detach().cpu().numpy()


def _build_random_indices_generator(shape):
    lists = [list(range(s)) for s in shape]
    combinations = list(product(*lists))
    shuffle(combinations)

    def generate():
        for indices in combinations:
            yield indices
        return None

    return generate, len(combinations)


def _generate_batch(data, generator, batch_size, window):
    result_center = []
    result_target = []
    for _ in range(batch_size):
        indices = generator()
        if indices is None:
            break
        sequence = data[indices[0]]
        result_row = []
        for i in range(window):
            result_row.append(sequence[indices[1] + i])
        result_target.append(result_row)
        result_center.append([sequence[indices[1] + window // 2]])
    return torch.tensor(result_center, dtype=torch.long), torch.tensor(result_target, dtype=torch.long)


def _generate_negative(batch_size, vocab_size):
    return torch.tensor([randint(0, vocab_size - 1) for _ in range(batch_size)], dtype=torch.long)


def train_skipgram(data, vocab_size, embed_size, window=2, negative_samples=3, batch_size=1024, iterations=10000,
                   lr=0.001, momentum=0.9,
                   silent=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = SkipGramModel(vocab_size, embed_size)
    model.train()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr, momentum=momentum)

    for i in range(iterations):
        if not silent:
            if i % 1000 == 0:
                print("Iteration:", i + 1)
        batch_generator = process_data(data, batch_size, window)

        avg_loss = 0.
        batch_center, batch_positive = next(batch_generator)
        batch_center = torch.tensor(batch_center, dtype=torch.long).to(device)
        batch_positive = torch.tensor(batch_positive, dtype=torch.long).to(device)
        results = [F.logsigmoid(model(batch_center, batch_positive)).view(-1)]

        for _ in range(negative_samples):
            batch_negative = _generate_negative(len(batch_center), vocab_size)
            batch_negative = torch.tensor(batch_negative, dtype=torch.long).to(device)
            results.append(F.logsigmoid(-1 * model(batch_center, batch_negative)).view(-1))

        optimizer.zero_grad()

        loss = -1 * torch.mean(torch.cat(tuple(results)))
        loss.backward()

        optimizer.step()

        avg_loss += loss.item()
        if not silent:
            if i % 1000 == 0:
                print("---| Loss", avg_loss)
    return model.get_embeddings()
