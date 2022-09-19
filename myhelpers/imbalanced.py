# from https://github.com/ufoym/imbalanced-dataset-sampler/

from builtins import isinstance
from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision
from sklearn.utils import class_weight
import numpy as np


def inverse_weights_by_labels(labels):
    df = pd.DataFrame()
    df["label"] = labels

    label_to_count = df["label"].value_counts()

    weights = 1.0 / label_to_count[df["label"]]

    return torch.DoubleTensor(weights.to_list())


# classses that have a high number of images have higher weights
# takes a dataset.targets
def get_class_weights(labels):
    # df = pd.DataFrame()
    # df["label"] = labels

    # label_to_count = df["label"].value_counts().sort_index()

    # return torch.tensor(label_to_count.values)

    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels),labels)
    class_weights = torch.tensor(class_weights,dtype=torch.float)
    return class_weights



class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        l = self._get_labels(dataset)
        df["label"] = [l[i] for i in self.indices]
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples