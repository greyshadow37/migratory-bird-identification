import torch
import torch.nn as nn
from torchvision import models

def create_model(n_classes, device):
    """
    Load EfficientNetB0 with ImageNet weights and modify the classifier.
    
    Args:
        n_classes (int): Number of output classes.
        device (torch.device): Device to move the model to (cuda or cpu).
    
    Returns:
        torch.nn.Module: Modified EfficientNetB0 model.
    """
    print(f"Loading EfficientNetB0 with ImageNet weights...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, n_classes)
    )
    print(f"Modified classifier: Dropout(0.3) -> Linear({num_features}, {n_classes})")
    return model.to(device)

def freeze_backbone(model):
    """
    Freeze the backbone (features) of the model, leaving the classifier trainable.
    
    Args:
        model (torch.nn.Module): The model to modify.
    """
    print("Freezing backbone, training classifier only...")
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

def unfreeze_finetune_layers(model):
    """
    Unfreeze stages 6 and 7 of EfficientNetB0 for fine-tuning, along with the classifier.
    
    Args:
        model (torch.nn.Module): The model to modify.
    """
    print("Unfreezing stages 6 and 7 of EfficientNetB0 for fine-tuning...")
    for name, param in model.named_parameters():
        if "features.6" in name or "features.7" in name:
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True