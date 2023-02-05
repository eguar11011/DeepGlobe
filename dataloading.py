import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
import warnings
warnings.filterwarnings("ignore")



#routes
base_dir:str = os.path.abspath("data")
train_dir:str = os.path.join(base_dir, "train")
test_dir:str = os.path.join(base_dir, "test")
valid_dir:str = os.path.join(base_dir, "valid")

#dataframes
filenames_df = pd.read_csv(os.path.join(base_dir, "metadata.csv"))
train_df = filenames_df[filenames_df['split']=='train']
class_dict_df = pd.read_csv(os.path.join(base_dir ,"class_dict.csv"))
# np.ndarray
img_ids: np.ndarray = filenames_df[filenames_df["split"] == "train"]["image_id"].values 




class MyDataset(Dataset):
    def __init__(self, img_dir:str,img_ids:np.ndarray, transform =None):
        self.img_dir= img_dir
        self.img_ids = img_ids
        self.transform = None
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_ids)

    def __getitem__(self, idx):
        'Generates one sample of data'
        ID:int = self.img_ids[idx]

        img_path = os.path.join(train_dir, f"{ID}_sat.jpg")
        img:np.ndarray = plt.imread(img_path)
        mask_path = os.path.join(train_dir, f"{ID}_mask.png")
        mask: np.ndarray = plt.imread(mask_path)


        return img, mask


if __name__ == "__main__":
        
    myset = MyDataset(train_df, img_ids)

    img_mask = DataLoader(myset, batch_size=5, shuffle=False)
    img, img_mask = next(iter(img_mask))


    print(img_mask[1].shape)