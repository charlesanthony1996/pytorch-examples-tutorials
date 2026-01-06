import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image

# positional encoding generator
class PositionaEncodingGenerator(nn.Module):
    def __init__(self, embed_dim):
        super(PositionaEncodingGenerator, self).__init__()
        self.depthwise_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)

    def forward(self, x):
        return self.depthwise_conv
    

# special multiscale adaptformer (sm-adpatformer)
class SMAdaptFormer(nn.Module):
    def __init__(self, embed_dim, reduction_dim):
        super(SMAdaptFormer, self).__init__()
        self.fc1 = nn.Linear(embed_dim, reduction_dim)
        self.fc2 = nn.Linear(reduction_dim, embed_dim)

        self.conv1x1 = nn.Conv2d(embed_dim, reduction_dim, kernel_size=1)
        self.conv3x3 = nn.Conv2d(embed_dim, reduction_dim, kernel_size=3, padding=1)


        self.dilated_conv12 = nn.Conv2d(embed_dim, reduction_dim, kernel_size=3, padding=12, dilation=12)
        self.dilated_conv24 = nn.Conv2d(embed_dim, reduction_dim, kernel_size=3, padding=24, dilation=24)
        self.dilated_conv36 = nn.Conv2d(embed_dim, reduction_dim, kernel_size=3, padding=36, dilation=36)

    def forward(self, x):
        x_fc = F.relu(self.fc1(x))
        x_fc = self.fc2(x_fc)

        x_conv1x1 = self.conv1x1(x)
        x_conv3x3 = self.conv3x3(x)

        x_dilated12 = self.dilated_conv12(x)
        x_dilated24 = self.dilated_conv24(x)
        x_dilated36 = self.dilated_conv36(x)

        x_concat = torch.cat([x_fc, x_conv1x1, x_conv3x3, x_dilated12, x_dilated24, x_dilated36], dim = 1)

        return x_concat
    
class GeneralizedSAM(nn.Module):
    def __init__(self, sam_model, embed_dim, reduction_dim):
        super(GeneralizedSAM, self).__init__()
        self.sam = sam_model
        self.peg = PositionaEncodingGenerator(embed_dim)
        self.sm_adaptformer = SMAdaptFormer(embed_dim, reduction_dim)

        # freezing sam layers if needed
        for param in self.sam.parameters():
            param.requires_grad = False


    def forward(self, x):
        x = self.peg(x)

        features = self.sam(x)

        fine_tuned_features = self.sm_adaptformer(features)

        segmentation_output = self.sam.decoder(fine_tuned_features)

        return segmentation_output
    
    

