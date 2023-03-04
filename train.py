import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
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
train_mask_dir: str = os.path.join(base_dir, "train_masks")

# dataframes
filenames_df = pd.read_csv(os.path.join(base_dir, "metadata.csv"))
train_df = filenames_df[filenames_df['split']=='train']
class_dict_df = pd.read_csv(os.path.join(base_dir ,"class_dict.csv"))
# np.ndarray
img_ids: np.ndarray = filenames_df[filenames_df["split"] == "train"]["image_id"].values 

def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
        #comment out if your model contains a sigmoid or equivalent activation layer
        y_pred = F.sigmoid(y_pred)       
        
        # skip the batch and class axis for calculating Dice score
        """axes = tuple(range(1, len(y_pred.shape)-1)) 
        numerator = 2. * np.sum(y_pred * y_true, axes)
        denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)"""
        class_preds += torch.stack(
    [
        (y_pred & y_true[:, c].unsqueeze(1)).sum(dim=[0, 2, 3])
        for c in range(7)
    ]).to("cpu")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


"""
Rutina de entrenamiento
"""
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        #print(X.shape, y.shape)
        X, y = X.squeeze(0), y.squeeze()
        # compute Prediction and loss
        pred = model(X)  # distribution per pixiel?
        #print(X.shape, y.shape)
        #loss = loss_fn(pred, y)
        loss = soft_dice_loss(y, pred) 
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
# datos

if __name__ == "__main__":

    myset = MyDataset(train_dir, train_mask_dir, img_ids)
    model = Unet()
    img_mask = DataLoader(myset, batch_size=3, shuffle=False)
    #img, img_mask = next(iter(img_mask))

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


