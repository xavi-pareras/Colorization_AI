import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
criterion = torch.nn.MSELoss()



def train_single_epoch(loader, model, optimizer, scheduler = None):

    model.train()
    losses = []

    for gray_images, color_images in loader:

        gray_images, color_images = gray_images.to(device), color_images.to(device)
        
        
        optimizer.zero_grad()
        output_images = model(gray_images)
        
        loss = criterion(output_images, color_images)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss)
        losses.append(loss.item())
        

    return np.mean(losses)


def eval_single_epoch( loader, model):
    losses = []
    with torch.no_grad():
        model.eval()
        for gray_images, color_images in loader:
            gray_images, color_images = gray_images.to(device), color_images.to(device)
            
            output_images = model(gray_images)
            loss = criterion(output_images, color_images)
            
            
           
            losses.append(loss.item())

    return np.mean(losses)


def train_model(config):

    transformToTensor = transforms.Compose([transforms.Resize(size=(250, 250)) ,transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    my_dataset = MyDataset('data\Grey','data\Color',
                         transform=transformToTensor)
    
  
    dataset_size = len(my_dataset)
    test_size = int(0.15 * dataset_size)
    val_size = int(0.15 * dataset_size)
    train_size = dataset_size - val_size - test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [train_size, test_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])
    my_model = MyModel(config).to(device)
    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in range(config["epochs"]):
        train_loss = train_single_epoch(
            train_loader, my_model, optimizer)
        print(f"Train Epoch {epoch} loss={train_loss:.5f} ")
        eval_loss = eval_single_epoch(
            test_loader, my_model)
        print(f"Test Epoch {epoch} loss={eval_loss:.5f}")

    return my_model


if __name__ == "__main__":
    config = {
        "lr": 0.001,
        "batch_size": 32,
        "epochs": 30,
    }

    train_model(config)
