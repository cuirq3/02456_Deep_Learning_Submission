import torch
import torch.nn as nn
import torch.nn.functional as F

class HelmetDetector(nn.Module):

    def __init__(self, input_channels, input_height, input_width, kernel_size):
        super(HelmetDetector, self).__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.kernel_size = kernel_size

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels,
                      out_channels=8,
                      kernel_size= self.kernel_size,
                      padding=int(self.kernel_size/2)),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(in_channels=8,
                      out_channels=10,
                      kernel_size=self.kernel_size,
                      padding=int(self.kernel_size/2)),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=10*(input_height//4)*(input_width//4),
                      out_features=32,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_features=32,
                      out_features=9,
                      bias=True),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        high_dim_feature = self.model(inputs)
        high_dim_feature = torch.flatten(high_dim_feature, 1)
        binary_feature = self.fc(high_dim_feature)
        return binary_feature
