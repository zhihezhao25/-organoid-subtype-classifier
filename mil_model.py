"""
Attention-MIL 多任务类器官亚型分类模型

设计目的：
- 不强依赖精确分割/切割。
- 把整张图片切成多个 patch。
- 模型自动学习哪些 patch 对亚型判断更重要。
- 三个数据集共享 backbone，但每个数据集使用自己的亚型分类头。
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class AttentionPooling(nn.Module):
    """对一组 patch 特征做 attention 加权聚合。"""

    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_features):
        scores = self.attention(patch_features).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        image_features = torch.sum(patch_features * weights.unsqueeze(-1), dim=1)
        return image_features, weights


class MILMultiTaskOrganoidClassifier(nn.Module):
    """弱切割友好的三数据集多任务分类模型。"""

    def __init__(self, tissue_configs, backbone_name="mobilenet_v3_small", pretrained=True, dropout=0.3):
        super().__init__()
        self.tissue_configs = tissue_configs
        self.tissue_types = list(tissue_configs.keys())
        self.backbone_name = backbone_name

        self.backbone, self.feature_dim = self._build_backbone(backbone_name, pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attention_pool = AttentionPooling(self.feature_dim)

        self.tissue_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, len(self.tissue_types)),
        )

        self.subtype_heads = nn.ModuleDict()
        for tissue_name, config in tissue_configs.items():
            self.subtype_heads[tissue_name] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, config["num_classes"]),
            )

        self._init_heads()

    def _build_backbone(self, backbone_name, pretrained):
        if backbone_name == "mobilenet_v3_small":
            weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            model = tv_models.mobilenet_v3_small(weights=weights)
            return model.features, 576
        if backbone_name == "mobilenet_v3_large":
            weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            model = tv_models.mobilenet_v3_large(weights=weights)
            return model.features, 960
        if backbone_name == "convnext_tiny":
            weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            model = tv_models.convnext_tiny(weights=weights)
            return model.features, 768
        if backbone_name == "efficientnet_b0":
            weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = tv_models.efficientnet_b0(weights=weights)
            return model.features, 1280
        raise ValueError(f"Unknown backbone: {backbone_name}")

    def _init_heads(self):
        heads = [self.tissue_head] + list(self.subtype_heads.values())
        for head in heads:
            for layer in head.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def encode_patches(self, patches):
        batch_size, num_patches, channels, height, width = patches.shape
        flat_patches = patches.view(batch_size * num_patches, channels, height, width)
        features = self.backbone(flat_patches)
        features = self.pool(features).flatten(1)
        return features.view(batch_size, num_patches, self.feature_dim)

    def forward(self, patches, tissue=None):
        patch_features = self.encode_patches(patches)
        image_features, attention_weights = self.attention_pool(patch_features)
        tissue_logits = self.tissue_head(image_features)

        if tissue is not None:
            subtype_logits = self.subtype_heads[tissue](image_features)
            return tissue_logits, subtype_logits, attention_weights

        subtype_logits = {
            tissue_name: head(image_features)
            for tissue_name, head in self.subtype_heads.items()
        }
        return tissue_logits, subtype_logits, attention_weights

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
