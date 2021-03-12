# import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, out_dim=10, in_channel=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(131072, out_dim, bias=True)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        # 	nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        # 	nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        # )
        # self.last = nn.Linear(256, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
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
