#https://github.com/adambielski/siamese-triplet

from torch.utils.data.sampler import BatchSampler
import numpy as np
import torch
from itertools import combinations
from torch import nn
import torch.nn.functional as F


## Metrics

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'



######################

# Sampler
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:								   
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
#################

# Selectors
def pdist(vectors):
    # print('vectors', vectors.shape)
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    # print('distance_matrix', distance_matrix.shape)
    return distance_matrix

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []
        
        # print('l', len(set(labels)))

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            # print('li', len(label_indices))
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)
            # print('ap', anchor_positives.shape)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                # print('distances', ap_distance, distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)], self.margin)
                # print('loss_values', loss_values)
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                # print('n', hard_negative)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    # print('n2', hard_negative)
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        # print('a', anchor_positives)
            
        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        # print('triplets', triplets.shape, triplets)
        triplets = np.array(triplets)
        # print('triplets 3.0', triplets.shape, triplets)
        # print('triplet_len', torch.LongTensor(triplets).shape)

        # Number of total possible triplets:
        #n_classes * (n_samples*n_fine /n_classes)C2

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)
       

##################

# Triplet loss

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector, device=None):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.device = device
        # self.hierarchyBased = hierarchyBased
        # self.triplet_layers_dic = triplet_layers_dic

    def forward(self, embeddings_, batch, layer_name, targetFromLayer):
        #Let's try normalizing the embedding
#         embeddings = embeddings/torch.norm(embeddings, dim=1).unsqueeze(1)
#         print('embeddings shape', embeddings.shape)
#         print('embeddings norm', torch.norm(embeddings, dim=1))
        
        embeddings = embeddings_[layer_name]
        embeddings = F.normalize(embeddings, p=2)

        # print(embeddings.shape)
        #TODO: This needs to be more sophisticated for phylogeny dataset
        # print('hello3', targetFromLayer.unusedLayersState)
        target = targetFromLayer.get_target_from_layerName(batch, layer_name, embeddings_)
        # print('hi', target, batch, layer_name, embeddings_)
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        # print('triplets 2', triplets.shape)

        if self.device is not None:
            triplets = triplets.cuda()
            
#         print('e', embeddings)
#         print('e1', embeddings[triplets[:, 0]])
#         print('e2', embeddings[triplets[:, 1]])
#         print('e3', embeddings[triplets[:, 2]])


        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        # if self.distance == "L2":
        #     ap_distances = ap_distances.pow(.5)
        #     an_distances = an_distances.pow(.5)
        
        
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)




###################



# Public
def get_tripletLossLoader(dataset, n_samples):
    kwargs = {}

    n_classes = len(dataset.csv_processor.getFineList())

    batch_sampler = BalancedBatchSampler(dataset.dataset.targets, n_classes=n_classes, n_samples=n_samples)
    return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)

def get_triplet_criterion(margin, selection_criterion, device):
    selector = None
    if selection_criterion=="semihard":
        selector = SemihardNegativeTripletSelector
    elif selection_criterion=="hard":
        selector = HardestNegativeTripletSelector
    elif selection_criterion=="random":
        selector = RandomNegativeTripletSelector
    else:
        raise NotImplementedError
 
    return OnlineTripletLoss(margin, selector(margin, device is None))

def get_triplet_output_mapping(batch, csv_processor):
    return csv_processor.get_triplet_output_mapping(batch)
    


