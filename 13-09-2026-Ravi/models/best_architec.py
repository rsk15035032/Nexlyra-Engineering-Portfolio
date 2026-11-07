import torch
import torch.nn as nn


class StrainVisionNet(nn.Module):

    def __init__(self,tabular_dim=5):
        super().__init__()

        # Spatial CNN Branch
        self.spatial_branch = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1),
                              nn.BatchNorm2d(32),
                              nn.LeakyReLU(inplace=True),
                              nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
                              nn.MaxPool2d(kernel_size=2,stride=2),
                              nn.Flatten())


        # Tabular MLP Branch
        self.tabular_branch = nn.Sequential(nn.Linear(tabular_dim,64),
                              nn.ReLU(inplace=True),
                              nn.Dropout(p=0.4),
                              nn.Linear(64,32),
                              nn.ReLU(inplace=True),
                              nn.Dropout(p=0.4))

        # Fusion
        self.fusion = nn.Sequential(nn.Linear(1024 + 32,128),
                      nn.ReLU(inplace=True),
                      nn.Dropout(p=0.4),
                      nn.Linear(128,1))

    def forward(self,spatial_x,tabular_x):
        spatial_features = (self.spatial_branch(spatial_x))
        tabular_features = (self.tabular_branch(tabular_x))

        combined_features = torch.cat([spatial_features,tabular_features],dim=1)
        logits = self.fusion(combined_features)

        return logits