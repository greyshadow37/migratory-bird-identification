import torch
import torch.nn as nn
from torchvision import models

def create_model(n_classes, device):
    model = models.resnet18(weights='IMAGENET1K_V1')  # Load pretrained ResNet18
    # Modify the fully connected layer for the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)
    return model.to(device)

def freeze_backbone(model):
    # Freeze all parameters except the fc layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

def unfreeze_finetune_layers(model):
    # Unfreeze layer4 (last residual block) and fc layer for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True