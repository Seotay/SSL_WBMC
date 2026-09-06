import torch.nn as nn
from torchvision.models import (resnet18, resnet34, resnet50, resnet101,
                                ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights)


class ResnetModel(nn.Module):
    def __init__(self, model_name="resnet18", num_classes=9, pretrained=True):
        super(ResnetModel, self).__init__()

        if model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
    
        elif model_name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = resnet34(weights=weights)
        
        elif model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)

        elif model_name == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = resnet101(weights=weights)
        else:
            raise ValueError("Unsupported model name")

        # Change the classification head to have num_classes output neurons
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(in_features=in_features, out_features=num_classes, bias=True))

    def forward(self, x):
        x = self.backbone(x)
        return x
    
