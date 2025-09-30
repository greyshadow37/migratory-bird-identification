import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# --------------------------
# Sequential Fusion Attention (same as before)
# --------------------------
class SFA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SFA, self).__init__()
        reduced_channels = max(in_channels // reduction_ratio, 4)

        # --- Channel attention ---
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        self.sigmoid_c = nn.Sigmoid()

        # --- Spatial attention ---
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid_s = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        mlp_out = self.mlp(avg_pool) + self.mlp(max_pool)
        channel_att = self.sigmoid_c(mlp_out).view(b, c, 1, 1)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid_s(self.conv_spatial(spatial_att))
        x = x * spatial_att

        return x


# --------------------------
# MobileNetV2 + SFA
# --------------------------
class SFAMobileNetV2(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(SFAMobileNetV2, self).__init__()
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        
        self.features = base_model.features
        last_channel = base_model.last_channel  # = 1280 for MobileNetV2
        self.sfa = SFA(in_channels=last_channel, reduction_ratio=8)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)   # backbone features
        x = self.sfa(x)        # SFA block
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # GAP before classifier
        x = self.classifier(x)
        return x


# --------------------------
# Helpers
# --------------------------
def create_model(num_classes, device):
    model = SFAMobileNetV2(num_classes, pretrained=True)
    return model.to(device)

def freeze_backbone(model):
    """
    Stage 1: Train classifier head only.
    - Freeze backbone (features)
    - Freeze SFA
    - Keep classifier trainable
    """
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def unfreeze_finetune_layers(model):
    """
    Stage 2: Fine-tuning.
    - Unfreeze last bottleneck block (features.18)
    - Unfreeze SFA
    - Keep classifier trainable
    - Everything else stays frozen
    """
    for name, param in model.named_parameters():
        if ("features.18" in name or 
            "sfa" in name or 
            "classifier" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
