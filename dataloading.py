import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import torch.nn.functional as F

import torchvision.transforms as transforms

import os
import warnings
warnings.filterwarnings("ignore")

"""
To do list

- Revisar el tipado de torch
- Normalización de imagenes
- Buscar lo del tipo de dato float
"""

#routes
base_dir:str = os.path.abspath("data")
train_dir:str = os.path.join(base_dir, "train")
test_dir:str = os.path.join(base_dir, "test")
valid_dir:str = os.path.join(base_dir, "valid")
train_mask_dir: str = os.path.join(base_dir, "train_masks")

#dataframes
filenames_df = pd.read_csv(os.path.join(base_dir, "metadata.csv"))
train_df = filenames_df[filenames_df['split']=='train']
class_dict_df = pd.read_csv(os.path.join(base_dir ,"class_dict.csv"))
# np.ndarray
img_ids: np.ndarray = filenames_df[filenames_df["split"] == "train"]["image_id"].values 



class MyDataset(Dataset):
    def __init__(self, img_dir:str, mask_dir:str,img_ids:np.ndarray, transform =None):
        self.img_dir = img_dir 
        self.mask_dir = mask_dir
        self.img_ids = img_ids
        self.transform = None
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_ids)

    def __getitem__(self, idx):
        'Generates one sample of data'
        ID:int = self.img_ids[idx]

        img_path = os.path.join(self.img_dir, f"{ID}_sat.jpg")
        img:torch.tensor = read_image(img_path).float()
        mask_path = os.path.join(self.mask_dir, f"{ID}_mask.png")
        mask:torch.tensor = read_image(mask_path)
        mask = mask.to(torch.int64)
        mask = torch.nn.functional.one_hot(mask, num_classes=7) #one hot


        img, mask = img.view(1, 3, 2448, 2448), mask.view(1,7,2448, 2448).float()
        rescaled_img = F.interpolate(img, size= (256,256), mode = 'bilinear', align_corners= False)
        
        rescaled_mask = F.interpolate(mask, size= (256,256), mode = 'bilinear', align_corners= False)

        return rescaled_img.squeeze(), torch.round(rescaled_mask.squeeze()).long()


if __name__ == "__main__":

    myset = MyDataset(train_dir, train_mask_dir, img_ids)
    img_mask = DataLoader(myset, batch_size=5, shuffle=False)
    img, mask = next(iter(img_mask))
    print(img[1].shape, mask.dtype , mask[1].shape)

