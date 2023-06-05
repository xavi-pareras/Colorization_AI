import torch
from torch.utils.data import DataLoader

from dataset import MyDataset
from model import MyModel
from utils import accuracy
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ray import tune

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(model, train_loader, optimizer):
    model.train()
    accs, losses = [], []
    for images, labels in train_loader:
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        acc = accuracy(outputs, labels)
        losses.append(loss.item())
        accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def eval_single_epoch(model, val_loader):
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            acc = accuracy(outputs, labels)
            losses.append(loss.item())
            accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def train_model(config):

    transformToTensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    my_dataset = MyDataset('/Users/orijf/OneDrive/Documentos/Visual Studio Main Folder/aidl-2023-spring-mlops/session-2/Chinese_MNIST_Data/data/data',
                           '/Users/orijf/OneDrive/Documentos/Visual Studio Main Folder/aidl-2023-spring-mlops/session-2/Chinese_MNIST_Data/chinese_mnist.csv', transform=transformToTensor)
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [10000, 2500, 2500])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    my_model = MyModel(config).to(device)

    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    for epoch in range(config["epochs"]):
        loss, acc = train_single_epoch(my_model, train_loader, optimizer)
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        val_loss, val_acc = eval_single_epoch(my_model, val_loader)
        print(f"Eval Epoch {epoch} loss={val_loss:.2f} acc={val_acc:.2f}")
        tune.report(val_loss=val_loss)
    
    test_loss, test_acc = eval_single_epoch(my_model, test_loader)
    print(f"Test loss={test_loss:.2f} acc={test_acc:.2f}")
    tune.report(test_loss=test_loss)

    return my_model


if __name__ == "__main__":

    analysis = tune.run(
        train_model,
        metric="val_loss",
        mode="min",
        num_samples=20,
        config = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 128]),
            "epochs": 10,
            "h1": tune.randint(10, 20),
            "h2": tune.randint(150, 170),
            "h3": tune.randint(100, 140),
            "h4": tune.randint(100, 120),
            "kernel_size1": 3,
            "kernel_size2": tune.choice([3,5]),
            "kernel_size3": tune.choice([3,5]),
        })

    print("Best hyperparameters found were: ", analysis.best_config)