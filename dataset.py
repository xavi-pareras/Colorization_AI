import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):

    def __init__(self, gray_directory, color_directory, transform=None):
        super().__init__()
        self.gray_directory = gray_directory
        self.color_directory = color_directory
        self.image_paths = []
        self.color_image_paths = []
        self.addDataFromPath(gray_directory)
        self.transform = transform
        
    



    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        return self.getImage(self.image_paths[idx]) , self.getImage(self.color_image_paths[idx])
        
        
        
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
 

