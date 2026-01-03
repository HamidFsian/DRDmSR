import torch
import torch.nn as nn
import torch.nn.functional as F

class CA(nn.Module):
    def __init__(self, filters, reducer=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(filters, filters // reducer, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(filters // reducer, filters, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RCAB(nn.Module):
    def __init__(self, filters, scale=0.1):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.ca = CA(filters)
        self.relu = nn.ReLU(inplace=True)
        self.scale = scale

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.ca(x)
        x = x * self.scale + residual
        return x


class RG(nn.Module):
    def __init__(self, filters, n_RCAB=20):
        super(RG, self).__init__()
        self.rcabs = nn.Sequential(*[RCAB(filters) for _ in range(n_RCAB)])
        self.conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.rcabs(x)
        x = self.conv(x)
        x = x + residual
        return x
