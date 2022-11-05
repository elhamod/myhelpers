import os
from CUB_dataloader import CUB_Dataset, datasetManager, generate_CUB_dataset_file, generate_CUB_190_dataset_file

params = {
    # Do not change for multi/hyperp experiments
    "image_path": "CUB_200_2011", # # Cifar has no subfolder
    "suffix":None, #"curated_30_50", #  None # used to get a subset cleaned_metadata file. set to None otherwise to use full metadata

    # newly added params
    "grayscale": False,
    
    # dataset
    "img_res": 32, #TODO: SHOULD NEVER BE HARCODED
    "augmented": False,

    # training
    "batchSize": 16, # Bigger is more stable (256)
    "learning_rate":0.00005, # 0.00005
    "numOfTrials":5,
    "fc_layers": 2,
    "modelType":"BB", #BB DISCO DSN HGNN HGNN_cat "HGNN_add" 
    "lambda": 0.01,
    "unsupervisedOnTest": False,
    "tl_model": "ResNet18", # Keep 'ResNet18', 'ResNet50', 'CIFAR', 'NIN'
    "link_layer": "avgpool", # layer name should be consistent with tl_model layer names. avgpool,"layer3"
    
    "adaptive_smoothing": False,
    "adaptive_lambda": 0.015, #optimized for time
    "adaptive_alpha": 0.8, #optimized for time
    
    "noSpeciesBackprop": False, 

    "phylogeny_loss": False,

}

experimentsPath="/raid/mridul/CUB_190_split/experiments/" # why are we specifying this ?
dataPath="/raid/mridul/CUB_190_split/official/"

manager = datasetManager('')

#Fix path:
params["image_path"] = os.path.join(dataPath, params["image_path"])
params

manager.updateParams(params)

# generate_CUB_dataset_file(params["image_path"])
# generate_CUB_190_dataset_file(params["image_path"])
train_loader, val_loader, test_loader = manager.getLoaders()

print(len(train_loader), len(val_loader), len(test_loader))
print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

print(train_loader.dataset[0])

print(train_loader.dataset.get_labels())

# print(train_loader.dataset[0])

