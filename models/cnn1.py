# import torch
import torch.nn as nn
# import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, out_dim=10, in_channel=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3,
                      stride=1, padding=0, bias=False),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3,
                      stride=1, padding=0, bias=False),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
        )

        self.linear = nn.Sequential(
            nn.Linear(1024, 1000, bias=False),
            nn.ReLU(),

            nn.Linear(1000, 1000, bias=False),
            nn.ReLU()
        )
        self.last = nn.Linear(1000, out_dim, bias=True)

    def features(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        # x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def cnn():
    return CNN()
