import numpy as np
import os
import random
from PIL import Image
from PIL import ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class MyDataset(data.Dataset):
    def __init__( self, data_dir, transform ):
        self.transform = transform

        self.imgs = []
        for root,dirs,filenames in os.walk( data_dir ):
            for filename in filenames:
                img_name = os.path.join( root, filename )
                self.imgs.append( img_name )

    def __getitem__( self, idx ):
        img = Image.open( self.imgs[ idx ] ).convert("RGB")
        
        img = self.transform(img)
        return img 
        
    def __len__( self ):
        return len( self.imgs ) 

#a=MyDataset( '/data/user/data1/lockeliu/learn/data/GAN/mnist', [] );
#print(a.__len__())
#b = a.__getitem__(11)
#print(b.size)
