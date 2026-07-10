"""
模型定义 —— 使用 torchvision 模型（从 PyTorch CDN 下载权重，不走 HuggingFace）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class GeMPooling(nn.Module):
    """Generalized Mean Pooling"""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p).flatten(1)


class OrganoidSubtypeClassifier(nn.Module):
    """
    类器官亚型分类
    输入: (B, 3, H, W)
    输出: (B, num_classes)
    """
    def __init__(self, num_classes=4, backbone_name="efficientnet_b3",
                 pretrained=True, dropout=0.3, freeze_stages=0):
        super().__init__()
        self.backbone_name = backbone_name

        # ---- ResNet ----
        if backbone_name.startswith("resnet"):
            w = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            full = tv_models.resnet50(weights=w)
            self.feature_dim = 2048
            self.backbone = nn.Sequential(
                full.conv1, full.bn1, full.relu, full.maxpool,
                full.layer1, full.layer2, full.layer3, full.layer4,
            )

        # ---- EfficientNet ----
        elif backbone_name.startswith("efficientnet"):
            if "b3" in backbone_name:
                w = tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
                full = tv_models.efficientnet_b3(weights=w)
                self.feature_dim = 1536
            elif "b0" in backbone_name:
                w = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                full = tv_models.efficientnet_b0(weights=w)
                self.feature_dim = 1280
            else:
                w = tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
                full = tv_models.efficientnet_b3(weights=w)
                self.feature_dim = 1536
            self.backbone = full.features

        # ---- ConvNeXt ----
        elif "convnext" in backbone_name:
            w = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            full = tv_models.convnext_tiny(weights=w)
            self.feature_dim = 768
            self.backbone = full.features

        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # ---- GeM Pooling ----
        self.gem_pool = GeMPooling(p=3.0)

        # ---- Classifier ----
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),          # LayerNorm 不会因 batch=1 报错
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._init_classifier()

        # ---- 冻结浅层 ----
        self._freeze_stages(freeze_stages)

    def _freeze_stages(self, n):
        """冻结 backbone 浅层"""
        if n <= 0:
            return
        children = list(self.backbone.children())
        for i in range(min(n, len(children))):
            for p in children[i].parameters():
                p.requires_grad = False

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        return self.gem_pool(self.backbone(x))

    def forward(self, x):
        return self.classifier(self.gem_pool(self.backbone(x)))

    def get_attention_map(self, x):
        features = self.backbone(x)
        features.retain_grad()
        logits = self.classifier(self.gem_pool(features))
        return features, logits


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    for name in ["efficientnet_b3", "resnet50", "convnext_tiny"]:
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        model = OrganoidSubtypeClassifier(num_classes=4, backbone_name=name, dropout=0.3)
        total, trainable = count_parameters(model)
        print(f"  Total: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")
        x = torch.randn(4, 3, 224, 224)
        logits = model(x)
        print(f"  Output: {logits.shape}")
