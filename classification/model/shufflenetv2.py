import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

def create_model(num_classes, device):
    # Load ShuffleNetV2 with pretrained weights
    model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    
    # Replace the classifier head for the specified number of classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    # Move model to the specified device
    model = model.to(device)
    return model

def freeze_backbone(model):
    # Freeze all layers except the classifier (fc)
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

def unfreeze_finetune_layers(model):
    # Unfreeze stage3, stage4, and classifier (fc) for fine-tuning #last feature block
    for name, param in model.named_parameters():
        if "stage3" in name or "stage4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False