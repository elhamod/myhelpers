import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision import transforms
from tqdm import tqdm
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib
from matplotlib.pyplot import imshow, show
import os
import pandas as pd
import seaborn as sns
import mpld3
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

MAX_DIMS_PCA=100

# Given a dataloader, a model, and an activation layer, it displays an images tsne
def get_tsne(dataloader, model, activation_layer, img_res, path, file_prefix, 
    legend_labels=['fine'], 
    cuda=None,
    sub_embedding_range=slice(0,None), 
    which_tsne_plots = ['standard', 'images', 'KNN', 'comimics', 'incorrect']):
    X = None

    # Get tsne
    a, n, _ = dataloader.dataset.toggle_image_loading(dataloader.dataset.augmentation_enabled, True)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image2 = batch['image']
        if cuda is not None:
            image2 = image2.cuda()
        # print(model.activations(image2).keys())
        features2 = model.activations(image2)[activation_layer][:, sub_embedding_range].detach().cpu()
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

    print("layer", activation_layer)

    if 'images' in which_tsne_plots:
        visualize_tsne_images(dataloader, tx, ty, img_res, path, file_prefix, model, cuda)

    if 'KNN' in which_tsne_plots:
        plot_phylo_KNN(dataloader, tx, ty, path, file_prefix)
    
    # This code is going to only work for butterflies:
    if 'comimics' in which_tsne_plots:
        try:
            plot_by_comimics(dataloader, tx, ty, path, file_prefix)
        except:
            print('Comimcs plotting does not apply to this case.')
            pass

    if 'standard' in which_tsne_plots:
        plot_tsne_dots(dataloader, tx, ty, path, file_prefix, legend_labels)
    
    if 'incorrect' in which_tsne_plots:
        plot_correct_incorrect(dataloader, tx, ty, path, file_prefix, model, cuda)

    

    show(block=False)
    print('--------')
    





def plot_tsne_dots(dataloader, tx, ty, path, file_prefix, legend_labels=['fine']):
    # Parse through the dataloader images
    file_names=[]

    labels = {}
    a, n, _ = dataloader.dataset.toggle_image_loading(dataloader.dataset.augmentation_enabled, True)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        label = {}
        for j in legend_labels:
            label[j] = batch[j]
        file_name = batch['fileName']
        for j in legend_labels:
            labels[j] = label[j] if j not in labels else torch.cat([labels[j], label[j]]).detach()
        file_names = file_names + file_name
    dataloader.dataset.toggle_image_loading(a, n)

    df = pd.DataFrame()
    df['tsne-x'] = tx
    df['tsne-y'] = ty
    for j in legend_labels:
        labels[j] = labels[j].tolist()
    
    # print(df)
    # print(set(fine_labels))
    # print(len(set(fine_labels)))

    if hasattr(dataloader.dataset, 'csv_processor'):
        for j in legend_labels:
            labels[j] = list(map(lambda x :dataloader.dataset.csv_processor.getList(j)[x], labels[j]))
        # print('ho', fine_labels)

    for j in legend_labels:
        df[j] = labels[j]

    for j in legend_labels:
        matplotlib.pyplot.figure(figsize=(16,10))
        sns_plot = sns.scatterplot(
            x="tsne-x", y="tsne-y",
            hue=j,
            palette=sns.color_palette("hls", len(set(labels[j]))),
            data=df,
            legend="full"
        )
        tooltip_label = [str(j) for i in range(len(labels))]
        tooltip = mpld3.plugins.PointLabelTooltip(sns_plot, labels=tooltip_label)
        fig = sns_plot.get_figure()
        mpld3.plugins.connect(fig, tooltip)
        fig.savefig(os.path.join(path, file_prefix+"_legend" + j +"_tsne_dots.png"))

def avg_distances(fine_label, indexes, dataset):
    csv_processor = dataset.csv_processor
    species_list = csv_processor.getFineList()
    dist = 0
    for i in indexes:
        lbl = dataset[i]['fine']
        dist = dist + csv_processor.tax.get_distance(species_list[fine_label], species_list[lbl])
    result = dist/len(indexes)
    return result



def plot_phylo_KNN(dataloader, tx, ty, path, file_prefix):
    # Parse through the dataloader images
    KNN_values = None
    a, n, _ = dataloader.dataset.toggle_image_loading(dataloader.dataset.augmentation_enabled, True)

    coord = np.hstack((np.array(tx).reshape(-1,1),np.array(ty).reshape(-1,1)))
    nbrs = NearestNeighbors(n_neighbors=5).fit(coord)
    _, indexes = nbrs.kneighbors(coord)
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        fine_label = batch['fine']
        batch_Width = len(fine_label)
        

        # image2 = batch['image']
        # if cuda is not None:
        #     image2 = image2.cuda()
        #     fine_label = fine_label.cuda()
        # print(fine_label, comimicsComponents)
        KNN_values_ = torch.tensor([avg_distances(x, indexes[i*batch_Width+i_], dataloader.dataset) for i_, x in  enumerate(fine_label)]) #fine_label.apply_(lambda x: )
        KNN_values = KNN_values_ if KNN_values is None else torch.cat([KNN_values, KNN_values_]).detach()
    
    dataloader.dataset.toggle_image_loading(a, n)

    df = pd.DataFrame()
    df['tsne-x'] = tx
    df['tsne-y'] = ty
    df['KNN_phylo_dist'] = KNN_values

    matplotlib.pyplot.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(
        x="tsne-x", y="tsne-y",
        hue='KNN_phylo_dist',
        palette="Oranges",
        data=df,
        legend="full"
    )

    norm = plt.Normalize(df['KNN_phylo_dist'].min(), df['KNN_phylo_dist'].max())
    sm = plt.cm.ScalarMappable(cmap="Oranges", norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    sns_plot.get_legend().remove()
    sns_plot.figure.colorbar(sm)

    fig = sns_plot.get_figure()
    fig.savefig(os.path.join(path, file_prefix+"_tsne_KNN_phylo.png"))  
 
def plot_by_comimics(dataloader, tx, ty, path, file_prefix):
    # Parse through the dataloader images
    comimics = None
    a, n, _ = dataloader.dataset.toggle_image_loading(dataloader.dataset.augmentation_enabled, True)

    # create comimics components
    comimicsComponents, subspecies_labels = dataloader.dataset.csv_processor.getComimicComponents()

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        fine_label = batch['fine']
        # image2 = batch['image']
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
    comimics_labels = comimics.tolist()
    comimics_labels = list(map(lambda x :subspecies_labels[comimicsComponents.tolist().index(comimicsComponents[x])], comimics_labels))
    df['comimcs_ring'] = comimics_labels

    matplotlib.pyplot.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(
        x="tsne-x", y="tsne-y",
        hue='comimcs_ring',
        palette=sns.color_palette("hls", len(set(comimics_labels))),
        data=df,
        legend="full"
    )
    fig = sns_plot.get_figure()
    fig.savefig(os.path.join(path, file_prefix+"_tsne_comimcs_ring.png"))


def plot_correct_incorrect(dataloader, tx, ty, path, file_prefix, model=None, cuda=None):
    # Parse through the dataloader images
    equality = None
    a, n, _ = dataloader.dataset.toggle_image_loading(dataloader.dataset.augmentation_enabled, True)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        fine_label = batch['fine']
        image2 = batch['image']
        if cuda is not None:
            image2 = image2.cuda()
            fine_label = fine_label.cuda()
        _, pred_ = torch.max(torch.nn.Softmax(dim=1)(model.activations(image2)['fine']),1)
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


def visualize_tsne_images(dataloader, tx, ty, img_res, path, file_prefix, model=None, cuda=None):
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
        _, pred_ = torch.max(torch.nn.Softmax(dim=1)(model.activations(image2)['fine']),1)
        preds = pred_ if preds is None else torch.cat([preds, pred_]).detach()
    dataloader.dataset.toggle_image_loading(a, n)

    # Construct the images
    width = 12000
    height = 9000
    max_dim = 150
    font = ImageFont.truetype(font='DejaVuSans.ttf', size=int(float(img_res) / 15))
    full_image = Image.new('RGB', (width, height))
    
    for img, x, y, file_name, fine_label, pred, coarse_label in tqdm(zip(images, tx, ty, file_names, fine_labels, preds, coarse_labels), total=tx.shape[0]):
        tile = transforms.ToPILImage()(img).convert("RGB")
        
        draw = ImageDraw.Draw(tile)
        draw.text((0, 0), file_name +  "\ncoarse: " + str(coarse_label.item()),(255,0,0), font=font)
        draw.text((0, int(img_res*0.8)), "true: " + str(fine_label.item())+"\npredicted: "+str(pred.item()), (255,0,0), font=font)
        
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), height - int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    matplotlib.pyplot.figure(figsize = (16,12))
    # full_image = full_image.transpose(Image.FLIP_TOP_BOTTOM)
    imshow(full_image)
    full_image.save(os.path.join(path, file_prefix+"_tsne_images.png")) 
