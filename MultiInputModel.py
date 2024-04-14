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
class MultiInputModel(nn.Module):
  def __init__(self, num_classes=7, num_demographic_feature=3):
    super(MultiInputModel, self).__init__()

    # load DenseNet
    self.densenet = models.densenet121(pretrained=True)

    # adjust last layer
    num_features = self.densenet.classifier.in_features
    self.densenet.classifier = nn.Linear(num_features, num_classes)

    # classifier for demographic features
    self.demographic_classifier = nn.Sequential(
        nn.Linear(num_demographic_feature, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes)
    )

    # final classifier to make final prediction
    self.final_classifier = nn.Linear(num_classes * 2, num_classes)

  def forward(self, image_input, age, localization, gender):
    image_input = image_input.float()
    age = age.float()
    localization = localization.float()
    gender = gender.float()

    # forward for densenet
    densenet_output = self.densenet(image_input)

    # Concatenate demographic features to feed into classifier
    demographic_features = torch.cat([age.unsqueeze(1), localization.unsqueeze(1), gender.unsqueeze(1)], dim=1)

    # Forward pass for demographic classifier
    demographic_output = self.demographic_classifier(demographic_features)

    # Concatenate outputs
    combined_output = torch.cat([densenet_output, demographic_output], dim=1)

    # Make final prediction
    final_output = self.final_classifier(combined_output)

    return final_output