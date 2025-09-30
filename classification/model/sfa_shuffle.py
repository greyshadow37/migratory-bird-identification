import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

# --------------------------
# Sequential Fusion Attention
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
# ShuffleNetV2 + SFA
# --------------------------
class SFAShuffleNetV2(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(SFAShuffleNetV2, self).__init__()
        base_model = shufflenet_v2_x1_0(
            weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Rebuild features manually (ShuffleNetV2 doesn’t expose `.features`)
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.maxpool,
            base_model.stage2,
            base_model.stage3,
            base_model.stage4,
            base_model.conv5,
        )

        last_channel = 1024  # ShuffleNetV2 x1.0 output dim
        self.sfa = SFA(in_channels=last_channel, reduction_ratio=8)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)               # backbone features
        x = self.sfa(x)                    # SFA block
        x = F.adaptive_avg_pool2d(x, 1)    # global average pooling
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --------------------------
# Helpers
# --------------------------
def create_model(num_classes, device):
    model = SFAShuffleNetV2(num_classes, pretrained=True)
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
    - Unfreeze stage3, stage4
    - Unfreeze SFA
    - Keep classifier trainable
    """
    for name, param in model.named_parameters():
        if ("features.3" in name or   # stage3
            "features.4" in name or   # stage4
            "sfa" in name or 
            "classifier" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
