import os
# from tkinter import N
import pandas as pd
import copy

def generate_file(dataset_folder_name):

    cub_reconciliation = pd.read_csv(os.path.join(dataset_folder_name,'CUB_taxon_reconciliation_corrected.csv'))

    # reading all the text files shared in the dataset
    images = pd.read_csv(os.path.join(dataset_folder_name,'images.txt'), sep=' ', header=None)
    images.rename(columns={0:'idx', 1: 'image_location'}, inplace=True)

    image_class_labels = pd.read_csv(os.path.join(dataset_folder_name,'image_class_labels.txt'), sep=' ', header=None)
    image_class_labels.rename(columns={0:'idx', 1: 'class_label'}, inplace=True)

    train_test_split = pd.read_csv(os.path.join(dataset_folder_name,'train_test_split.txt'), sep=' ', header=None)
    train_test_split.rename(columns={0:'idx', 1: 'train_test_split'}, inplace=True)

    # merging all the file information into single df across the index in the files
    dataset_information = images.merge(image_class_labels,on='idx', how='inner').merge(train_test_split,on='idx', how='inner')

    # getting class name from the image location
    dataset_information['class'] = dataset_information['image_location'].str.split('/',expand=True)[0]
    dataset_information['class'] = dataset_information['class'].str.split('.',expand=True)[1]

    # flagging classes for which phlygeny information is present. 1-present, 0-not present
    dataset_information['within_190'] = dataset_information['class'].isin(list(cub_reconciliation['English'])).astype(int)
    dataset_information.to_csv(os.path.join(dataset_folder_name,'CUB_dataset_information.csv'), index=False)

    classes_not_present = list(dataset_information[dataset_information['within_190']==0]['class'].unique())
    print(classes_not_present)

    # keeping only for which phlyogeny is present and re-indexing
    cub_190 = dataset_information[dataset_information['within_190']==1]
    cub_190.reset_index(inplace=True)
    cub_190.drop(columns='index', inplace=True)
    cub_190.reset_index(inplace=True)
    cub_190['index'] += 1

    cub_190.rename(columns={'idx' : 'old_index', 'index': 'new_index'}, inplace=True)
    cub_190.to_csv(os.path.join(dataset_folder_name,'CUB_190_dataset_information.csv'), index=False)

    # TODO: do we need to change the class label as well?



    return None

dataset_folder_name='/raid/mridul/CUB_190/official/CUB_200_2011'
generate_file(dataset_folder_name)