import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def create_model(num_classes, device):
    # Load MobileNetV2 with pretrained weights
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Replace the classifier head for the specified number of classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
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
    # Unfreeze the last few layers for fine-tuning
    # For MobileNetV2, unfreeze the last block (features[18]) and classifier
    for name, param in model.named_parameters():
        if "features.18" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False