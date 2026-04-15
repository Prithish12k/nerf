import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_dim, dir_dim):
        super().__init__()
        self.layers1 = nn.ModuleList(
            [nn.Linear(input_dim, 256)] + [nn.Linear(256, 256) for _ in range(3)]
        )

        self.skip_layer = nn.Linear(256+input_dim, 256)

        self.layers2 = nn.ModuleList([
            nn.Linear(256, 256) for _ in range(3)
        ])

        self.sigma_layer = nn.Linear(256, 1)
        self.feature_layer = nn.Linear(256, 256)

        self.color1 = nn.Linear(256+dir_dim, 256)
        self.color2 = nn.Linear(256, 3)

    def forward(self, x, d):
        input_x = x

        for layer in self.layers1:
            x = torch.relu(layer(x))

        x = torch.cat([x, input_x], -1)
        x = torch.relu(self.skip_layer(x))

        for layer in self.layers2:
            x = torch.relu(layer(x))

        sigma = F.softplus(self.sigma_layer(x))
        x = self.feature_layer(x)
        x = torch.cat([x, d], -1)

        col = torch.sigmoid(self.color2(torch.relu(self.color1(x))))

        return col, sigma