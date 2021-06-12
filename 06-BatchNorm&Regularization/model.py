import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, norm):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(10) if norm=="BN" else nn.GroupNorm(2, 10) if norm=="GN" else nn.LayerNorm([10, 28, 28]),
            nn.Dropout2d(0.01),

            nn.Conv2d(10, 10, 3),
            nn.ReLU(),
            nn.BatchNorm2d(10) if norm=="BN" else nn.GroupNorm(2, 10) if norm=="GN" else nn.LayerNorm([10, 26, 26]),
            nn.Dropout2d(0.01),

            nn.Conv2d(10, 10, 3),
            nn.ReLU(),
            nn.BatchNorm2d(10) if norm=="BN" else nn.GroupNorm(2, 10) if norm=="GN" else nn.LayerNorm([10, 24, 24]),
            nn.Dropout2d(0.01),

            nn.MaxPool2d(2, 2)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(10) if norm=="BN" else nn.GroupNorm(2, 10) if norm=="GN" else nn.LayerNorm([10, 12, 12]),
            nn.Dropout2d(0.01),

            nn.Conv2d(10, 10, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(10) if norm=="BN" else nn.GroupNorm(2, 10) if norm=="GN" else nn.LayerNorm([10, 12, 12]),
            nn.Dropout2d(0.01),

            nn.Conv2d(10, 10, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(10) if norm=="BN" else nn.GroupNorm(2, 10) if norm=="GN" else nn.LayerNorm([10, 12, 12]),
            nn.Dropout2d(0.01),

            nn.MaxPool2d(2, 2)
        )
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16) if norm=="BN" else nn.GroupNorm(4, 16) if norm=="GN" else nn.LayerNorm([16, 6, 6]),
            nn.Dropout2d(0.01),

            nn.Conv2d(16, 15, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(15) if norm=="BN" else nn.GroupNorm(3, 15) if norm=="GN" else nn.LayerNorm([15, 6, 6]),
            nn.Dropout2d(0.01),
            
            nn.Conv2d(15, 10, 3, padding=1),
            nn.AvgPool2d(6)
        )

        self.fcblock = nn.Linear(10, 10)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x.view(-1, 10)
        x = self.fcblock(x)
        return F.log_softmax(x, dim=1)