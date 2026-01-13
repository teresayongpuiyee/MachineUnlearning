"""
Source: https://github.com/weiaicunzai/pytorch-cifar100
"""

from torch import nn
import torch
from model.resnet import ResNet, BasicBlock, BottleNeck
import torch.nn.functional as F


def ResNet18(num_classes, input_channels):
    """return a ResNet 18 object"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channel= input_channels)


def ResNet34(num_classes, input_channels):
    """return a ResNet 34 object"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes= num_classes, input_channel= input_channels)


def ResNet50(num_classes, input_channels):
    """return a ResNet 50 object"""
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes= num_classes, input_channel= input_channels)


def ResNet101(num_classes, input_channels):
    """return a ResNet 101 object"""
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes= num_classes, input_channel= input_channels)


def ResNet152(num_classes, input_channels):
    """return a ResNet 152 object"""
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes= num_classes, input_channel= input_channels)


class MLP(nn.Module):
    def __init__(
        self,
        input_features,
        hidden_layer1,
        hidden_layer2,
        out_features
    ):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden_layer1)
        self.f_connected2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.out = nn.Linear(hidden_layer2, out_features)

    def feature_extractor(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        return x

    def classifier_head(self, x):
        x = self.out(x)
        return x

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier_head(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # Assuming input image size is 32x32
        self.fc2 = nn.Linear(128, num_classes)

    def feature_extractor(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x
    
    def classifier_head(self, x):
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier_head(x)
        return x

# Linear regression in pytorch
class LRTorchNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim
    ):
        super(LRTorchNet, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class AttackMLP(nn.Module):
    def __init__(self, input_size, hidden_1, hidden_2, output_size):
        super(AttackMLP, self).__init__()
        
        # Define the layers
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout for regularization (preventing attack model overfitting)
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, output_size) # Output raw logits (no sigmoid here)
        )

    def forward(self, x):
        # x is the feature vector (batch_size, 512)
        logits = self.layer_stack(x)
        return logits