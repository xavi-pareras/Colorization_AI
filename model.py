import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5),  nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2, stride = 2))


         # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size=5, stride = 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size=5, stride =2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size=6),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        
        return x