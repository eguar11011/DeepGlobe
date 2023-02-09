import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F

"""import sys 
print(sys.path.append('../'))
#from dataloading import MyDataset"""

from dataloading import MyDataset


""""
Revisar sobre la inforación perdida.
"""

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
out_sz=(256,256)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        conv3x3->ReLU->conv3x3->ReLU 
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3) # padding
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x)))) # Esta es la secuencia

class Encoder(nn.Module):
    def __init__(self, chs = (3,64,128, 256, 512, 1024)): # Aumento de los canales
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1])for i in range(len(chs)-1)]) # parece la lista generada del modelo
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class Unet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=3, retain_dim=True, out_sz=(256,256)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, out_sz, mode = 'bilinear', align_corners= False) 
        return out.squeeze()


"""
Log del programa
"""

if __name__ == "__main__":


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



    myset = MyDataset(train_df, img_ids)
    model = Unet()
    img_mask = DataLoader(myset, batch_size=1, shuffle=False)
    f, v = False, True

    test1= v
    test2= f
    test3= f # necesita del test2
    test4= f

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    if test1:
        enc_block = Block(1,64)
        x = torch.randn(1,572,572)
        print(enc_block(x).shape)

    if test2:
        encoder = Encoder()
        #input image
        x = torch.randn(1,3,572,572)
        ftrs = encoder(x)
        for ftr in ftrs: print(ftr.shape)

    if test3:
        decoder = Decoder()
        x = torch.randn(1, 1024, 28, 28)
        print(decoder(x, ftrs[::-1][1:]).shape)

    if test4:
        unet = Unet()
        x    = torch.randn(1, 3, 572, 572)
        print(unet(x).shape)
