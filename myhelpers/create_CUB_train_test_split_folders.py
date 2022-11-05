import os
import shutil
import pandas as pd
from tqdm import tqdm

train_files = pd.read_csv('/raid/mridul/CUB_190_split/official/CUB_200_2011/train.txt', header=None)
test_files = pd.read_csv('/raid/mridul/CUB_190_split/official/CUB_200_2011/test.txt', header=None)

base_path = '/raid/mridul/CUB_190_split/official/CUB_200_2011/images'
train_path = '/raid/mridul/CUB_190_split/official/CUB_200_2011/train'
test_path = '/raid/mridul/CUB_190_split/official/CUB_200_2011/test'

# test_img = '200.Common_Yellowthroat/Common_Yellowthroat_0004_190606.jpg'
# old_path = os.path.join(base_path,test_img)
# new_path = os.path.join(train_path,test_img)

# os.makedirs(os.path.join(train_path,'200.Common_Yellowthroat'))
# shutil.copy(old_path, new_path)

for _,row in tqdm(train_files.iterrows(), total=train_files.shape[0]):
    old_path = os.path.join(base_path, row[0])
    new_path = os.path.join(train_path, row[0])
    sub_dir = row[0].split('/')[0]
    sub_dir_path = os.path.join(train_path, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    shutil.copy(old_path, new_path)

for _,row in tqdm(test_files.iterrows(), total=test_files.shape[0]):
    old_path = os.path.join(base_path, row[0])
    new_path = os.path.join(test_path, row[0])
    sub_dir = row[0].split('/')[0]
    sub_dir_path = os.path.join(test_path, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    shutil.copy(old_path, new_path)



print('done')