import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import os

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from dataloading import MyDataset
from U_net import Unet, Decoder, Encoder, Block

# routes
base_dir:str = os.path.abspath("data")
train_dir:str = os.path.join(base_dir, "train")
test_dir:str = os.path.join(base_dir, "test")
valid_dir:str = os.path.join(base_dir, "valid")

# dataframes
filenames_df = pd.read_csv(os.path.join(base_dir, "metadata.csv"))
train_df = filenames_df[filenames_df['split']=='train']
class_dict_df = pd.read_csv(os.path.join(base_dir ,"class_dict.csv"))
# np.ndarray
img_ids: np.ndarray = filenames_df[filenames_df["split"] == "train"]["image_id"].values 


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


"""
Rutina de entrenamiento
"""
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        # compute Prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
# datos

if __name__ == "__main__":

    myset = MyDataset(train_df, img_ids)
    model = Unet()
    img_mask = DataLoader(myset, batch_size=1, shuffle=False)
    # img, img_mask = next(iter(img_mask))

    # parametros

    learning_rate = 1e-3
    batch_size = 64
    epochs = 1

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr = learning_rate)


    # Bucle de entrenamiento 

    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------ ")
        train_loop(img_mask, model, loss_fn, optimizer)

        
    print("Done!")