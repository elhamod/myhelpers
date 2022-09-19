import os
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torchvision import transforms, datasets
import pandas as pd
import copy

from myhelpers.read_write import Reader_writer
from myhelpers.dataset_normalization import dataset_normalization

        
class CUB_Dataset(Dataset):
    def __init__(self, type_, params, normalizer=None, verbose=False):
        self.imageDimension = params["img_res"]
        self.n_channels = 3
        self.data_root, self.suffix  = getParams(params)
        self.augmentation_enabled = params['augmented']
        self.normalizer = None
        self.normalization_enabled = True
        self.pad = False
        self.composedTransforms = None
        self.csv_processor = self
        self.fineToCoarseMatrix = None
        self.type_ = type_
        self.grayscale = params["grayscale"]

        data_root_suffix = os.path.join(self.data_root, self.suffix, type_)  
            
        print("Loading dataset...")       
        print(data_root_suffix)     

        self.dataset = datasets.ImageFolder(data_root_suffix, transform=transforms.Compose(self.getTransforms()), target_transform=None)
        self.mapFileNameToIndex = {} # This dictionary will make it easy to find the information of an image by its file name.
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = dataset_normalization(data_root_suffix, self.dataset, res=params['img_res']).getTransform()[0]
        self.RGBmean = [round(self.normalizer.mean[0]*255), round(self.normalizer.mean[1]*255), round(self.normalizer.mean[2]*255)]
        self.pad = True

        # We don't have the genus for CUB dataset. so, we are just going to have species.
        self.fileNames = x[b'filenames']
            
        metadata = unpickle(os.path.join(self.data_root,'cifar-100-python/meta'))
        self.coarse_index_list = metadata[b'coarse_label_names']

        # Create transfroms
        # Toggle beforehand so we could create the normalization transform. Then toggle back.
        if self.normalizer is None:
            augmentation, normalization, _ = self.toggle_image_loading(augmentation=False, normalization=False)   
            print("CIFAR normalization")
            # Cifar normalization: 
            self.normalizer = [transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
            self.toggle_image_loading(augmentation, normalization)

        
        # build distance_matrix
        self.distance_matrix = None
        self.build_taxonomy()
    
    def getTransforms(self):

        transformsList = []
        if self.augmentation_enabled:
            transformsList = transformsList + [
                transforms.RandomCrop(size=(32, 32), padding=4),
                transforms.RandomHorizontalFlip()
            ]

        transformsList = transformsList + [transforms.ToTensor()]
            
        if self.normalization_enabled:
            transformsList = transformsList + self.normalizer
        
        return transformsList

    def __len__(self):
        return len(self.dataset)
    
    # The list of fine/coarse names
    def getFineList(self):
        return self.dataset.classes
    def getCoarseList(self):
        return self.coarse_index_list

    def getCoarseLabel(self, fileName):
        idx = self.fileNames.index(fileName)
        return self.dataset[idx][1]['coarse']
    def getFineLabel(self, fileName):
        idx = self.fileNames.index(fileName)
        return self.dataset[idx][1]['fine']


    def getCoarseFromFine(self, fine):
        fineIndex = self.dataset.classes.index(fine)
        for coarse in self.coarse_to_fine:
            fineInCoarse = self.coarse_to_fine[coarse]
            if fineIndex in fineInCoarse:
                return self.coarse_index_list[coarse]
    
    # Returns a list of fine_index that belong to a coarse
    def getFineWithinCoarse(self, coarse):
        return list(map(lambda x: self.dataset.classes[x], self.coarse_to_fine[self.coarse_index_list.index(coarse)]))
    

    def getFineToCoarseMatrix(self):
        if self.fineToCoarseMatrix is None:
            self.fineToCoarseMatrix = torch.zeros(len(self.getFineList()), len(self.getCoarseList()))
            for fine_name in self.getFineList():
                coarse_name = self.getCoarseFromFine(fine_name)
                fine_index = self.getFineList().index(fine_name)
                coarse_index = self.getCoarseList().index(coarse_name)
                self.fineToCoarseMatrix[fine_index][coarse_index] = 1
        return self.fineToCoarseMatrix
    
    def toggle_image_loading(self, augmentation, normalization, pad=None):
        old = (self.augmentation_enabled, self.normalization_enabled, None)
        self.augmentation_enabled = augmentation
        self.normalization_enabled = normalization
        # print(self.type_, self.augmentation_enabled)
        self.transforms = None
        return old
        
    def getTargetTransform(self, target):
        return {
            'fine': target,
            'coarse': self.coarse_index_list.index(self.getCoarseFromFine(self.dataset.classes[target]))
        }

    def get_target_from_layerName(self, batch, layer_name, hierarchyBased=True, z_triplet=None, triplet_layers_dic=['layer2', 'layer3']):
        result = None
        first_layer = triplet_layers_dic[0]
        second_layer = triplet_layers_dic[1]
        
        if layer_name == first_layer:
            result = batch['coarse' if hierarchyBased==True else 'fine'] 
        elif layer_name == second_layer:
            result = batch['fine']
            
        return result

    
    def __getitem__(self, idx):       
        if self.transforms is None:
            # print('transform', self.type_, self.augmentation_enabled)
            self.transforms = self.getTransforms()
            # print(self.transforms)
            # print('---')
            self.composedTransforms = transforms.Compose(self.transforms)
            self.dataset.transform = self.composedTransforms
            
        image, target = self.dataset[idx]
        # if torch.cuda.is_available():
        #     image = image.cuda()

        return {'image': image, 
                'fine': target['fine'], 
                'fileName': self.fileNames[idx], #TODO Is this full name?
                'coarse': target['coarse'],} 

    def build_taxonomy(self):
        fineList = self.getFineList()

        self.distance_matrix = torch.zeros(len(fineList), len(fineList))
        for i, species_i in enumerate(fineList):
            for j, species_j in enumerate(fineList):
                if i == j:
                    self.distance_matrix[i, j] = 0
                elif self.getCoarseFromFine(species_i) == self.getCoarseFromFine(species_j): 
                    self.distance_matrix[i, j] = 2
                else:
                    self.distance_matrix[i, j] = 4





# Given a model and a dataset, get an example image where trueLabel=trueIndex where predictedIndex=expectedIndex
# The key param decides if we are looking at "fine" or "coarse"
# def getExamples(model, dataset, trueIndex, expectedIndex, key="fine", num_of_examples=1):
#     result = []
    
#     # get all examples trueLabel=fineIndex
#     if key == "fine":
#         name = dataset.getFineOfIndex(trueIndex)
#         examples = dataset.getFineIndices(name)
#     else:
#         examples = []
#         fine_set = dataset.getFineWithinCoarse(dataset.getCoarseList()[trueIndex])
#         for i in fine_set:
#             examples = examples + dataset.getFineIndices(i)  
            

#     # Find an example that predictedLabel=expectedIndex
# #     random.shuffle(examples)
#     for example in examples:
#         augmentation, normalization = dataset.toggle_image_loading(augmentation=False, normalization=False)
#         image = dataset[example]['image'].unsqueeze(0)
#         dataset.toggle_image_loading(augmentation, normalization)
#         predictionImage = dataset[example]['image'].unsqueeze(0)
#         predictedIndex = model(predictionImage)
#         if isinstance(predictedIndex,dict):
#             predictedIndex = predictedIndex[key]
#         predictedIndex = torch.max(predictedIndex.data, 1)[1].cpu().detach().numpy()[0]
#         if (predictedIndex == expectedIndex):
#             image = image.squeeze()
#             predictionImage = predictionImage.squeeze()
#             result.append((image, predictionImage))
#             if len(result) == num_of_examples:
#                 break

#     return result



def getIndices(data_root, train=True):
    fileName = 'train_train.txt' if train else 'train_val.txt'
    read_file = pd.read_csv(os.path.join(data_root,fileName), header = None, delimiter=' ')
    read_file.columns = ['index','target']
    return read_file['index'].tolist()

def getParams(params):
    data_root = params["image_path"]
    suffix = str(params["suffix"]) if ("suffix" in params and params["suffix"] is not None) else ""    
    return data_root, suffix
    
class datasetManager:
    def __init__(self, experimentName, verbose=False):
        self.verbose = verbose
        self.suffix = None
        self.dataset_train = None
        self.dataset_test = None
        self.experimentName = experimentName
        self.reset()

    def get_traintestsplit(self):
        self.train = []
        self.test = []

        with open(os.path.join(self.dataset_folder_name, 'train_test_split.txt'), 'r')as f:
            for i in f.readlines():
                idx, flag = i.strip().split(' ')
                if int(flag):
                    self.train.append(idx)
                else:
                    self.test.append(idx)
    
    def reset(self):
        self.dataset_train = None
        self.dataset_test = None
        self.train_loader = None
        self.validation_loader =  None
        self.test_loader = None
    
    def updateParams(self, params):
        self.reset()
        self.params = params
        self.data_root, self.suffix = getParams(params)
        self.experiment_folder_name = os.path.join(self.data_root, self.suffix, self.experimentName)
        self.dataset_folder_name = self.experiment_folder_name
        self.get_traintestsplit()
        
    def getDataset(self):
        if self.dataset_train is None:
            print("Creating dataset...")
            self.dataset_train = CifarDataset("train", self.params, verbose=self.verbose)
            self.dataset_test = CifarDataset("test", self.params, verbose=self.verbose)
            print("Creating dataset... Done.")
        return self.dataset_train, self.dataset_test

    # Creates the train/val/test dataloaders out of the dataset 
    def getLoaders(self):
        if self.dataset_train is None:
            self.getDataset()

        batchSize = self.params["batchSize"]

        # Save which indices are used for train vs validation
        index_fileNames = [trainingIndexFileName, valIndexFileName]
        saved_index_file = os.path.join(self.dataset_folder_name, valIndexFileName)
        loader_indices = []
        if not os.path.exists(saved_index_file):
            
            train_indices = getIndices(self.data_root)
            val_indices = getIndices(self.data_root, False)

            print("train/val = ", len(train_indices),len(val_indices))
            # save indices
            loader_indices = [train_indices, val_indices]
            for i, name in enumerate(index_fileNames):
                fullFileName = os.path.join(self.dataset_folder_name, name)
                reader_writer = Reader_writer('pkl', self.dataset_folder_name, fullFileName)
                reader_writer.writeFile(loader_indices[i])

        else:
            # load the pickles
            print("Loading saved indices...")
            for i, name in enumerate(index_fileNames): 
                fullFileName = os.path.join(self.dataset_folder_name, name)
                reader_writer = Reader_writer('pkl', self.dataset_folder_name, fullFileName)    
                loader_indices.append(reader_writer.readFile())


        # create samplers
        train_sampler = SubsetRandomSampler(loader_indices[0])
        valid_sampler = SubsetRandomSampler(loader_indices[1])

        # create data loaders.
        print("Creating loaders...")
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, sampler=train_sampler, batch_size=batchSize)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset_train, sampler=valid_sampler, batch_size=batchSize)
        self.test_loader = torch.utils.data.DataLoader(copy.copy(self.dataset_test), batch_size=batchSize)
        self.test_loader.dataset.toggle_image_loading(augmentation=False, normalization=self.dataset_test.normalization_enabled) # Needed so we always get the same prediction accuracy 
        print("Creating loaders... Done.")
            
        return self.train_loader, self.validation_loader, self.test_loader




def generate_CUB_dataset_file(dataset_folder_name):
    imgs_ = {'train': [], 'test': []}
    labels_ = {'train': [], 'test': []}
    if not os.path.exists(os.path.join(dataset_folder_name, 'train.txt')):
        # train test splitting
        train = []
        test = []
        with open(os.path.join(dataset_folder_name, 'train_test_split.txt'), 'r')as f:
            for i in f.readlines():
                idx, flag = i.strip().split(' ')
                if int(flag):
                    train.append(idx)
                else:
                    test.append(idx)

        # get image paths
        images = {}
        with open(os.path.join(dataset_folder_name, 'images.txt'), 'r')as f:
            for i in f.readlines():
                idx, path = i.strip().split(' ')
                images[idx] = path


        # get labels
        labels = {}
        with open(os.path.join(dataset_folder_name, 'image_class_labels.txt'), 'r')as f:
            for i in f.readlines():
                idx, label = i.strip().split(' ')
                # label in annotation is start with 1
                labels[idx] = int(label) - 1


        # put all together in two files
        with open(os.path.join(dataset_folder_name, 'train.txt'), 'w')as f:
            for idx in train:
                f.write('{},{}\n'.format(images[idx], labels[idx]))
                imgs_['train'].append(images[idx])
                labels_['train'].append(labels[idx])


        with open(os.path.join(dataset_folder_name, 'test.txt'), 'w')as f:
            for idx in test:
                f.write('{},{}\n'.format(images[idx], labels[idx]))
                imgs_['test'].append(images[idx])
                labels_['test'].append(labels[idx])

    else:
        with open(os.path.join(dataset_folder_name, 'train.txt'), 'r')as f:
            for i in f.readlines():
                img, label = i.strip().split(',')
                imgs_['train'].append(img)
                labels_['train'].append(label)
        
        with open(os.path.join(dataset_folder_name, 'test.txt'), 'r')as f:
            for i in f.readlines():
                img, label = i.strip().split(',')
                imgs_['test'].append(img)
                labels_['test'].append(label)
    
    return imgs_, labels_
    