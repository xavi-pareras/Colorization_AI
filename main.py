import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms

from dataset import MyDataset
from model import Unet
from utils import  save_model, init_model
from PIL import Image
from ray import tune



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
criterion = torch.nn.MSELoss()
SIZE = 256
optmizeHyperParameters = False
cpuThreads = 24


def run_single_epoch(loader, model, optimizer=None, scheduler=None):
    losses = []
  
    for data in loader:
        l_images = data["L"]
        ab_images = data["ab"]
        
        l_images, ab_images = l_images.to(device), ab_images.to(device)
        
        if model.training: 
            optimizer.zero_grad()

        output_images = model(l_images)
        
        loss = criterion(output_images, ab_images)
        if model.training:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss)
        
        losses.append(loss.item())
        

    return np.mean(losses)


def train_single_epoch(loader, model, optimizer, scheduler = None):

    model.train()
    return run_single_epoch(loader, model, optimizer, scheduler)



def eval_single_epoch( loader, model):
    
    with torch.no_grad():
        model.eval()
        return run_single_epoch(loader, model)

   


def train_model(config):

    transformToTensor = transforms.Compose([
        transforms.Resize((SIZE, SIZE),  Image.BICUBIC) ,
        transforms.ToTensor(), 
        transforms.RandomHorizontalFlip()])
    my_dataset = MyDataset('/Users/orijf/OneDrive/Documentos/VisualStudioMainFolder/Colorization/data/Grey','/Users/orijf/OneDrive/Documentos/VisualStudioMainFolder/Colorization/data/Color',
                         transform=transformToTensor)
    
  
    dataset_size = len(my_dataset)
    test_size = int(0.15 * dataset_size)
    val_size = int(0.15 * dataset_size)
    train_size = dataset_size - val_size - test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [train_size, test_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])
    model = init_model(Unet(config), device)
    optimizer = optim.Adam(model.parameters(), config["lr"])
    
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    min_loss = 1
    for epoch in range(config["epochs"]):
        train_loss = train_single_epoch(
            train_loader, model, optimizer)
        print(f"Train Epoch {epoch} loss={train_loss:.5f} ")
        eval_loss = eval_single_epoch(
            test_loader, model)
        print(f"Test Epoch {epoch} loss={eval_loss:.5f}")
       

        if  optmizeHyperParameters:
            tune.report(val_loss=eval_loss)
        else:
            if(eval_loss<min_loss):
                min_loss = eval_loss
                save_model(model, 'model_weights')
        

    


if __name__ == "__main__":
        if optmizeHyperParameters:
            analysis = tune.run(
                train_model,
                metric="val_loss",
                mode="min",
                num_samples=12,
                resources_per_trial={
                    "cpu": 2},
                stop={
                 "training_iteration": 10000000,
                },
                config = {
                    "lr": tune.loguniform(1e-4, 1e-2),
                    "batch_size": tune.choice([32, 64, 128]),
                    "epochs": 10,              
                    "input_c": 1,
                    "output_c": 2, 
                    "n_down": tune.choice([2, 4, 8]),
                    "num_filters": tune.choice([32, 64]),
                })
            print("Best hyperparameters found were: ", analysis.best_config)
        else:
  
            config = {
                "lr": 0.00044749588937525244,
                "batch_size": 64,
                "epochs": 30, 
                "input_c": 1,
                "output_c": 2, 
                "n_down": 2,
                "num_filters": 64            
            }


            if cpuThreads > 1:
                 analysis = tune.run(
                train_model,
                metric="val_loss",
                mode="min",
                num_samples=12,
                resources_per_trial={
                    "cpu": cpuThreads},
                config = config)
            else:
                train_model(config)
  
    
