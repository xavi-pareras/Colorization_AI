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




def train_single_epoch(loader, model, optimizer, scheduler = None):

    model.train()
    accuracies, losses = [], []

    for images, labels in loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss)
        
        acc = accuracy(outputs, labels)
        losses.append(loss.item())
        accuracies.append(acc.item())

    return np.mean(losses), np.mean(accuracies)


def eval_single_epoch( loader, model):
    accuracies, losses = [], []
    with torch.no_grad():
        model.eval()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            acc = accuracy(outputs, labels)
            losses.append(loss.item())
            accuracies.append(acc.item())

    return np.mean(losses), np.mean(accuracies)


def train_model(config):

    transformToTensor = transforms.Compose([transforms.Resize(size=(250, 250)) ,transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    my_dataset = MyDataset('data\grey',
                         transform=transformToTensor)
    
  
    dataset_size = len(my_dataset)
    test_size = int(0.15 * dataset_size)
    val_size = int(0.15 * dataset_size)
    train_size = dataset_size - val_size - test_size
    print(test_size)
    print(val_size)
    print(train_size)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [train_size, test_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    for images, labels in my_dataset:
        print(images.shape)
   


if __name__ == "__main__":
    config = {
        "lr": 0.0008129571910666688,
        "batch_size": 32,
        "epochs": 30,
        "h1": 15,
        "h2": 160,
        "h3": 125,
        "h4": 109,
        "kernel_size1":3,
        "kernel_size2": 5,
        "kernel_size3": 3,
    }

    train_model(config)
