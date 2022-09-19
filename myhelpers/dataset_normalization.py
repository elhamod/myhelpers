import torch
from torchvision import transforms
from .read_write import JSON_reader_writer
from tqdm import tqdm

class dataset_normalization():
    def __init__(self, dir_name, dataset, res=224):
        self.dataset_size = len(dataset)
        self.res = res
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        self.dir_name = dir_name

        self.reader_writer = JSON_reader_writer(dir_name, "dataset_normlization.json")
        self.read_file = self.reader_writer.readFile()

        self.mean = None
        self.std = None

        self.transform = None
        self.inv_transform = None
        
    def getTransform(self):
        if self.read_file is None:
            self.mean = 0.
            self.std = 0.
            nb_samples = 0.
            
            # Calculate the median per channel
            with tqdm(total=self.dataset_size, desc="getting statistics") as bar:
                for data in self.dataloader:
                    data  = data[0]
                    batch_samples = data.size(0)
                    data = data.view(batch_samples, data.size(1), -1)
                    self.mean += data.mean(-1).sum(0)
                    self.std += data.std(-1).sum(0)
                    nb_samples += batch_samples
                    bar.update()

            self.mean /= nb_samples
            self.std /= nb_samples

            self.mean = self.mean.numpy().tolist()
            self.std = self.std.numpy().tolist()

            print('dataset has a mean: {0} and std: {1}'.format(self.mean, self.std) )

            self.reader_writer.writeFile({
                "mean": self.mean,
                "std": self.std
            })
        else:
            self.mean = self.read_file["mean"]
            self.std = self.read_file["std"]

        self.transform = transforms.Normalize(self.mean, self.std)


        return [self.transform]

    def getDenormalizeTransform(self):
        if self.transform is None:
            self.getTransform()
        
        return [transforms.Normalize([-self.transform.mean[i] / self.transform.std[i] for i, _ in enumerate(self.transform.mean)], [1 / self.transform.std[i] for i, _ in enumerate(self.transform.mean)])]