import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision import transforms
from tqdm import tqdm
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib
from matplotlib.pyplot import imshow
import os
import pandas as pd
import seaborn as sns

MAX_DIMS_PCA=100

# Given a dataloader, a model, and an activation layer, it displays an images tsne
def get_tsne(dataloader, model, activation_layer, img_res, path, file_prefix, legend_labels=['fine'], cuda=None):
    X = None

    # Get tsne
    a, n, _ = dataloader.dataset.toggle_image_loading(dataloader.dataset.augmentation_enabled, True)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image2 = batch['image']
        if cuda is not None:
            image2 = image2.cuda()
        features2 = model(image2)[activation_layer].detach().cpu()
        features2 = features2.reshape(features2.shape[0], -1)

        # Calculate distance for each pair.
        X = features2 if X is None else torch.cat([X, features2]).detach()
    dataloader.dataset.toggle_image_loading(a, n)

    if X.shape[1]>MAX_DIMS_PCA:
        pca = PCA(n_components=MAX_DIMS_PCA)
        X = pca.fit_transform(X)

    tsne = TSNE(n_components=2, learning_rate=150, verbose=2).fit_transform(X)


    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    visualize_tsne_images(dataloader, tx, ty, img_res, path, file_prefix, legend_labels, model, cuda)
    
    # This code is going to only work for butterflies:
    try:
        plot_by_comimics(dataloader, tx, ty, img_res, path, file_prefix, legend_labels, model, cuda)
    except:
        pass

    plot_tsne_dots(dataloader, tx, ty, img_res, path, file_prefix, legend_labels, model, cuda)
    
    plot_correct_incorrect(dataloader, tx, ty, img_res, path, file_prefix, legend_labels, model, cuda)
    





def plot_tsne_dots(dataloader, tx, ty, img_res, path, file_prefix, legend_labels=['fine'], model=None, cuda=None):
    # Parse through the dataloader images
    fine_labels=None
    coarse_labels=None
    file_names=[]

    a, n, _ = dataloader.dataset.toggle_image_loading(dataloader.dataset.augmentation_enabled, True)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        fine_label = batch['fine']
        coarse_label = batch['coarse']
        file_name = batch['fileName']
        fine_labels = fine_label if fine_labels is None else torch.cat([fine_labels, fine_label]).detach()
        coarse_labels = coarse_label if coarse_labels is None else torch.cat([coarse_labels, coarse_label]).detach()
        file_names = file_names + file_name
    dataloader.dataset.toggle_image_loading(a, n)

    df = pd.DataFrame()
    df['tsne-x'] = tx
    df['tsne-y'] = ty
    fine_labels = fine_labels.tolist()
    coarse_labels = coarse_labels.tolist()
    
    # print(df)
    # print(set(fine_labels))
    # print(len(set(fine_labels)))

    if hasattr(dataloader.dataset, 'csv_processor'):
        # print('hi', fine_labels)
        fine_labels = list(map(lambda x :dataloader.dataset.csv_processor.getFineList()[x], fine_labels))
        coarse_labels = list(map(lambda x :dataloader.dataset.csv_processor.getCoarseList()[x], coarse_labels))
        # print('ho', fine_labels)

    df['coarse'] = coarse_labels
    df['fine'] = fine_labels

    for legend_label in legend_labels:
        matplotlib.pyplot.figure(figsize=(16,10))
        sns_plot = sns.scatterplot(
            x="tsne-x", y="tsne-y",
            hue=legend_label,
            palette=sns.color_palette("hls", len(set(fine_labels if legend_label=='fine' else coarse_labels))),
            data=df,
            legend="full"
        )
        fig = sns_plot.get_figure()
        fig.savefig(os.path.join(path, file_prefix+"_legend" + legend_label +"_tsne_dots.png"))




def plot_by_comimics(dataloader, tx, ty, img_res, path, file_prefix, legend_labels=['fine'], model=None, cuda=None):
    # Parse through the dataloader images
    comimics = None
    a, n, _ = dataloader.dataset.toggle_image_loading(dataloader.dataset.augmentation_enabled, True)

    # create comimics components
    comimicsComponents, subspecies_labels = dataloader.dataset.csv_processor.getComimicComponents()

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        fine_label = batch['fine']
        image2 = batch['image']
        # if cuda is not None:
        #     image2 = image2.cuda()
        #     fine_label = fine_label.cuda()
        # print(fine_label, comimicsComponents)
        comimics_ = fine_label.apply_(lambda x: comimicsComponents[x])
        comimics = comimics_ if comimics is None else torch.cat([comimics, comimics_]).detach()
    
    dataloader.dataset.toggle_image_loading(a, n)

    df = pd.DataFrame()
    df['tsne-x'] = tx
    df['tsne-y'] = ty
    fine_labels = comimics.tolist()
    fine_labels = list(map(lambda x :subspecies_labels[comimicsComponents.tolist().index(comimicsComponents[x])], fine_labels))
    df['comimcs_ring'] = fine_labels

    matplotlib.pyplot.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(
        x="tsne-x", y="tsne-y",
        hue='comimcs_ring',
        palette=sns.color_palette("hls", len(set(fine_labels))),
        data=df,
        legend="full"
    )
    fig = sns_plot.get_figure()
    fig.savefig(os.path.join(path, file_prefix+"_tsne_comimcs_ring.png"))


def plot_correct_incorrect(dataloader, tx, ty, img_res, path, file_prefix, legend_labels=['fine'], model=None, cuda=None):
    # Parse through the dataloader images
    equality = None
    a, n, _ = dataloader.dataset.toggle_image_loading(dataloader.dataset.augmentation_enabled, True)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        fine_label = batch['fine']
        image2 = batch['image']
        if cuda is not None:
            image2 = image2.cuda()
            fine_label = fine_label.cuda()
        _, pred_ = torch.max(torch.nn.Softmax(dim=1)(model(image2)['fine']),1)
        equality_ = torch.eq(pred_, fine_label)
        equality = equality_ if equality is None else torch.cat([equality, equality_]).detach()
        
    dataloader.dataset.toggle_image_loading(a, n)

    df = pd.DataFrame()
    df['tsne-x'] = tx
    df['tsne-y'] = ty
    fine_labels = equality.tolist()
    df['isCorrect'] = fine_labels
    # print(df)
    # print(set(fine_labels))
    # print(len(set(fine_labels)))

    matplotlib.pyplot.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(
        x="tsne-x", y="tsne-y",
        hue='isCorrect',
        palette=sns.color_palette("hls", len(set(fine_labels))),
        data=df,
        legend="full"
    )
    fig = sns_plot.get_figure()
    fig.savefig(os.path.join(path, file_prefix+"_tsne_correctPrediction.png"))


def visualize_tsne_images(dataloader, tx, ty, img_res, path, file_prefix, legend_labels=None, model=None, cuda=None):
    # Parse through the dataloader images
    images = None
    fine_labels=None
    preds=None
    coarse_labels=None
    file_names=[]

    a, n, _ = dataloader.dataset.toggle_image_loading(dataloader.dataset.augmentation_enabled, False)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image2 = batch['image']
        fine_label = batch['fine']
        coarse_label = batch['coarse']
        file_name = batch['fileName']
        images = image2 if images is None else torch.cat([images, image2]).detach()
        fine_labels = fine_label if fine_labels is None else torch.cat([fine_labels, fine_label]).detach()
        coarse_labels = coarse_label if coarse_labels is None else torch.cat([coarse_labels, coarse_label]).detach()
        file_names = file_names + file_name
    dataloader.dataset.toggle_image_loading(a, n)
    a, n, _ = dataloader.dataset.toggle_image_loading(dataloader.dataset.augmentation_enabled, True)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image2 = batch['image']
        if cuda is not None:
            image2 = image2.cuda()
        _, pred_ = torch.max(torch.nn.Softmax(dim=1)(model(image2)['fine']),1)
        preds = pred_ if preds is None else torch.cat([preds, pred_]).detach()
    dataloader.dataset.toggle_image_loading(a, n)

    # Construct the images
    width = 4000
    height = 3000
    max_dim = 100
    font = ImageFont.truetype(font='DejaVuSans.ttf', size=int(float(img_res) / 15))
    full_image = Image.new('RGB', (width, height))
    
    for img, x, y, file_name, fine_label, pred, coarse_label in tqdm(zip(images, tx, ty, file_names, fine_labels, preds, coarse_labels), total=tx.shape[0]):
        tile = transforms.ToPILImage()(img).convert("RGB")
        
        draw = ImageDraw.Draw(tile)
        draw.text((0, 0),file_name +  ", " + str(fine_label.item())+"-"+str(pred.item())+  ", " + str(coarse_label.item()),(255,0,0), font=font)
        
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    matplotlib.pyplot.figure(figsize = (16,12))
    imshow(full_image)
    full_image.save(os.path.join(path, file_prefix+"_tsne_images.png")) 