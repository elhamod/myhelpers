import matplotlib.pyplot as plt
import numpy as np
import torchvision as tv
import pandas as pd
import torch
from tqdm import tqdm


import math

# given a dataset, a model, an activation name, and an input, find K the closest image in that space
# It also prints histogram of distances from all images of dataset to that image
def getClosestImageFromDataloader(loader, model, activation_layer, indx, topk=2, cuda=None):  
    image=loader.dataset[indx]['image'].unsqueeze(0)

    img_per_row = 6

    dist_func = torch.nn.MSELoss(reduction='none')
    hist_list = []
    
    f, axarr = plt.subplots(math.ceil(topk/img_per_row),img_per_row, squeeze=False, figsize=(15,15))
   
    if cuda:
        image = image.cuda()
    
    # get activation of image
    features = model(image)[activation_layer]
    distance = None
    
    # loop through loader
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        image2 = batch['image']
        if cuda:
            image2 = image2.cuda()
        features2 = model(image2)[activation_layer]
        features2_1 = features2.reshape(features2.shape[0], -1)
        feature1_1 = torch.cat(features2.shape[0]*[features]).reshape(features2_1.shape[0], -1)
    
        # Calculate distance for each pair.
        # print('hello', features2.shape, feature1_1.shape)
        d = dist_func(feature1_1, features2_1).detach()
        temp = torch.sqrt(torch.sum(d, 1))
        distance =  temp if distance is None else torch.cat([distance, temp]) 
    
    if cuda:
        distance = distance.cpu()
    # print('dist', distance)
    hist_list = list(distance.numpy()) 
    # print('hist', hist_list)
    distance = distance.reshape(-1)
    inv_distance = 1/distance
    
    _, topk_distances_indices = inv_distance.topk(topk, 0, True, True)
    
    # print information about best one
    dataset = loader.dataset
    a, n, _ = loader.dataset.toggle_image_loading(dataset.augmentation_enabled, False)
    for k, index in tqdm(enumerate(topk_distances_indices), total=len(topk_distances_indices)):
        batch = dataset[index]
        img = batch['image']
        npimg = img.numpy()   # convert from tensor
        axarr[math.floor(k/img_per_row),k%img_per_row].imshow(np.transpose(npimg, (1, 2, 0)))
        fileName = batch['fileName']
        fineName = dataset.csv_processor.getFineList()[batch['fine']]
        coarseName = dataset.csv_processor.getCoarseList()[batch['coarse']]
        d = distance[index].item()

        string = fileName + "\n"
        string = string + fineName + "\n"
        string = string + coarseName + "\n"
        string = string +"d = " + str(round(d, 2))
        axarr[math.floor(k/img_per_row), k%img_per_row].set_title(string)
        
    loader.dataset.toggle_image_loading(a, n)

    plt.figure()
    plt.hist(hist_list, 20)
    plt.title("Histogram of distances")
    



# Given a dataset, the index of the example, and the model's prediction.
# It plots the image and its details and returns a dataframe of the predictions sorted
def showExample(dataset, index, predictions):    
    prediction = predictions[index, :]
    a, n, _ = dataset.toggle_image_loading(dataset.augmentation_enabled, False)
    entry =dataset[index]
    dataset.toggle_image_loading(a,n)
    imshow(tv.utils.make_grid(entry['image']))
    fileName = entry['fileName']
    truth_fine = dataset.csv_processor.getFineList()[entry['fine']]
    truth_coarse = dataset.csv_processor.getCoarseList()[entry['coarse']]
    print('File name', fileName)
    print('Truth', truth_fine+"_"+str(entry['fine'].item()), truth_coarse)
    print('Sorted predictions')
    df = pd.DataFrame(columns=["Top", "fine label", "coarse label", "model's prob of fine label"])
#     print(entry['fine'], prediction)
    prediction_sorted, prediction_indices = torch.sort(prediction, descending=True)
#     print(prediction_sorted, prediction_indices)
    for i, pred in enumerate(prediction_sorted):
        pred_fine = dataset.csv_processor.getFineList()[prediction_indices[i].item()]
        pred_coarse = dataset.csv_processor.getCoarseFromFine(pred_fine)
        pred_prob = pred.item()
        df = df.append({
         "Top": i,
         "fine label":  pred_fine + "_" + str(prediction_indices[i].item()),
         "coarse label":  pred_coarse,
         "model's prob of fine label":  pred_prob,
        }, ignore_index=True)
    return df


# Given a tensor image, ot shows it.
def imshow(img):
    img = img #/ 2 + 0.5   # unnormalize
    npimg = img.numpy()   # convert from tensor
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.show()
    