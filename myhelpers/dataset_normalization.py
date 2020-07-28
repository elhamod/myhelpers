import torch
from torchvision import transforms
from tqdm import tqdm

class dataset_normalization():
    def __init__(self, dataset, res=224):
        self.dataset_size = len(dataset)
        self.res = res
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
    def getTransform(self):
        self.mean = 0.
        self.std = 0.
        nb_samples = 0.
        
        # Calculate the median per channel
        with tqdm(total=self.dataset_size, desc="getting statistics") as bar:
            for data in self.dataloader:
                data  = data["image"]
                batch_samples = data.size(0)
                data = data.view(batch_samples, data.size(1), -1)
                self.mean += data.mean(-1).sum(0)
                self.std += data.std(-1).sum(0)
                nb_samples += batch_samples
                bar.update()

        self.mean /= nb_samples
        self.std /= nb_samples
        print('dataset has a mean: {0} and std: {1}'.format(self.mean, self.std) )
        return [transforms.Normalize(self.mean, self.std)]