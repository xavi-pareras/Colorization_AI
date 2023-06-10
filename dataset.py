import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import torchvision.transforms as transforms

class MyDataset(Dataset):

    def __init__(self, gray_directory, color_directory, transform=None):
        super().__init__()
        self.gray_directory = gray_directory
        self.color_directory = color_directory
        self.image_paths = []
        self.color_image_paths = []
        self.addDataFromPath(gray_directory)
        self.transform = transform
        self.image_channel_axis = self.getImage(self.color_image_paths[0]).shape.index(3)
        self.channel_axis = 2
        
        
    



    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img = Image.open(self.color_image_paths[idx]).convert("RGB")
        img = self.transform(img)
        img = np.array(img)
        img = np.moveaxis(img, self.image_channel_axis, self.channel_axis)
        img_lab = rgb2lab(img, channel_axis = self.channel_axis).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
        return {'L': L, 'ab': ab}
        
        
        
    def getImage(self, path):
        image = Image.open(path)
        
        if self.transform is None: 
            return image 
        else:
            return self.transform(image)
        

    def addDataFromPath(self,path):
        if(os.path.isdir(path)):
            for filename in os.listdir(path):
                self.addDataFromPath(os.path.join(path, filename))
        else:
            self.addImagePath(path)
   
    def addImagePath(self,path):
        
        color_path = os.path.join(self.color_directory, os.path.relpath(path, start = self.gray_directory)) 
        if os.path.exists(color_path):
            self.image_paths.append(path)
            self.color_image_paths.append(color_path)
        #self.labels.append(os.path.split(os.path.dirname(path))[1])
 

