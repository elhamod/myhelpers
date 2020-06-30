import torch
from torchvision import transforms

class dataset_normalization():
    def __init__(self, dataset):
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        
    def getTransform(self):
        mean = 0.
        std = 0.
        nb_samples = 0.
        
        # Calculate the median per channel
        for data in self.dataloader:
            data  = data["image"]
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(-1).sum(0)
            std += data.std(-1).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        print('dataset has a mean: {0} and std: {1}'.format(mean, std) )
        return [transforms.Normalize(mean, std)]