import math
from typing import Tuple

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def generate_mock_data(
    num_samples: int = 50_000, latent_width: int = 1024, class_count: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Generate mock data to test the evaluation harness

    :param num_samples: the number of samples to generate
    :type num_samples: int
    :param latent_width: the width of the sample. default 1024
    :type latent_width: int
    :param class_count: number of classes to expect. default 100
    :type class_count: int
    :return: a tuple of mock embeddings and mock labels
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    '''

    # @smallcrawler
    # > for now you can just assume I will give you a tensor of 50,000 random
    # > vectors of size 1024 and another tensor of 50,000 integers between 0 and
    # > 100.

    # embeddings are easy
    embeddings = torch.rand((num_samples, latent_width))

    # labels require flooring
    labels = torch.rand((num_samples)) * class_count
    labels = torch.floor(labels).to(torch.int)

    return embeddings, labels


class RepQualityLinear(nn.Module):
    '''Linear network to judge representation quality'''

    class_count: int
    latent_width: int

    def __init__(self, latent_width: int = 1024, class_count: int = 100):
        '''
        Set up the linear layer accuracy predictor. Should be used with a
        softmax cross entropy loss function (nn.CrossEntropyLoss)

        :param latent_width: the width of the latent. default 1024
        :type latent_width: int
        :param class_count: number of classes to predict. default 100
        :type class_count: int
        '''
        super().__init__()

        self.class_count = class_count
        self.latent_width = latent_width

        self.linear = nn.Sequential(
            nn.Linear(latent_width, class_count),
        )

    def forward(self, x):
        return self.linear(x)


def prepare_datasets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    test_set_size: float,
) -> Tuple[TensorDataset, TensorDataset]:
    '''

    :param test_set_size: Size of the test set as a percentage
    :type test_set_size: float
    :return: 
    :rtype: Tuple[TensorDataset, TensorDataset]
    '''
    test_set_size = math.floor(embeddings.shape[0] * test_set_size)

    # create the datasets and loaders
    train_dataset = TensorDataset(embeddings[test_set_size:, :], labels[test_set_size:])
    test_dataset = TensorDataset(embeddings[:test_set_size, :], labels[:test_set_size])

    return train_dataset, test_dataset


def train_linear_layer(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_count: int = 100,
    test_set_size: float = 0.10,
    device: str = 'gpu',
) -> Tuple[float, RepQualityLinear]:
    '''
    Create and train a simple linear layer to evaluate embedder performance.

    :param embeddings: latent embeddings to evaluate
    :type embeddings: torch.Tensor
    :param labels: labels to fit to
    :type labels: torch.Tensor
    :param class_count: Number of classes
    :type class_count: int
    :param test_set_size: Size of the test set as a percentage
    :type test_set_size: float
    :param device: device to run on
    :type device: str
    :return: the accuracy and the trained model
    :rtype: Tuple[float, RepQualityLinear]
    '''
    latent_width = embeddings.shape[1]

    # create the datasets and loaders
    train_dataset, test_dataset = prepare_datasets(embeddings, labels, test_set_size)
    train_loader = DataLoader(train_dataset, batch_size=64)
    test_loader = DataLoader(train_dataset, batch_size=64)

    # create the model
    model = RepQualityLinear(latent_width, class_count)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # train loop
    print('Beginning train...')
    model.to(device)
    model.train()

    for batch, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            print(f'    loss: {loss:>7f} batch:{batch}/{len(train_dataset)//train_loader.batch_size}')

    # test loop
    model.eval()
    size = len(test_dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0

    print('Evaluating model...')
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test error:\n    Accuracy: {(correct * 100):>0.1f}%, Avg loss: {test_loss:>8f}')

    return correct, model


def train_knn(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    test_set_size: float = 0.10,
    n_neighbors: int = 5
) -> Tuple[float, KNeighborsClassifier]:
    train_dataset, test_dataset = prepare_datasets(embeddings, labels, test_set_size)
    
    train_X = np.array(train_dataset.tensors[0])
    train_y = np.array(train_dataset.tensors[1])
    test_X  = np.array(test_dataset.tensors[0])
    test_y  = np.array(test_dataset.tensors[1])

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors
    )


    print('Creating knn...')
    knn.fit(train_X, train_y)

    print('Evaluating knn...')
    accuracy = knn.score(test_X, test_y)

    print(f'Test error:\n    Accuracy: {(accuracy*100):>0.2f}%')

    return accuracy, knn


if __name__ == '__main__':
    embeddings, labels = generate_mock_data(
        num_samples=50_000, latent_width=1024, class_count=100
    )

    ll_accuracy, _  = train_linear_layer(embeddings, labels, device='mps')

    knn_accuracy, _ = train_knn(embeddings, labels)

    print('accuracies')
    print(f'    linear layer accuracy: {ll_accuracy * 100:>.2f}%')
    print(f'    knn accuracy: {knn_accuracy * 100:>.2f}%')
