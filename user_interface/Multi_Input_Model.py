# imports
import pandas as pd
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import timeit
import torchvision
# import torchsummary
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
# from torchsummary import summary
# need this to load the model
# changed to take in one hot encoded inputs
# need this to load the model
# changed to take in one hot encoded inputs
from efficientnet_pytorch import EfficientNet

class MultiInputModel(nn.Module):
    def __init__(self, num_classes=7, num_localization_classes=6, num_gender_classes=2):
        super(MultiInputModel, self).__init__()

        # Load EfficientNet (pretrained)
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        # modify last layer
        num_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(num_features, num_classes)

        # Define a small neural network for demographic inputs
        # keeping it small because these features are less complex and lower in dimensionality than the images.
        # also to prevent overfitting where model learns from noise instead.
        demographic_input_size = num_localization_classes + num_gender_classes + 1  # total size of concatenated demographic inputs
        self.demographic_classifier = nn.Sequential(
            nn.Linear(demographic_input_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

        # Final classifier to make final prediction
        self.final_classifier = nn.Linear(num_classes * 2, num_classes)

    def forward(self, image_input, age, localization, gender):
    # Forward pass for image input
        print("Age shape:", age.shape)
        print("Localization shape:", localization.shape)
        print("Gender shape:", gender.shape)
        image_features = self.efficientnet(image_input)
        
        # Concatenate demographic inputs
        demographic_features = torch.cat([age, localization, gender], dim=1)


        # Forward pass for demographic classifier
        demographic_output = self.demographic_classifier(demographic_features)

        # Combine image features and demographic output
        combined_features = torch.cat([image_features, demographic_output], dim=1)

        # Final classification
        final_output = self.final_classifier(combined_features)

        return final_output
