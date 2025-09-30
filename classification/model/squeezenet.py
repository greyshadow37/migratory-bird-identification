import torch
import torch.nn as nn
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights

def create_model(num_classes, device):
    # Load SqueezeNet with pretrained weights
    model = squeezenet1_0(weights=SqueezeNet1_0_Weights.IMAGENET1K_V1)
    
    # Replace the classifier head for the specified number of classes
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Conv2d(512, num_classes, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1)
    )
    
    # Move model to the specified device
    model = model.to(device)
    return model

def freeze_backbone(model):
    # Freeze all layers except the classifier
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

def unfreeze_finetune_layers(model):
    # Unfreeze fire8 (features.10), fire9 (features.11-12), and classifier for fine-tuning
    for name, param in model.named_parameters():
        if "features.10" in name or "features.11" in name or "features.12" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False