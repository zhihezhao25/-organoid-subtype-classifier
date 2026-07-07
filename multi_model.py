"""
Multi-Organoid Classifier — 统一的多类器官亚型分类模型

架构:
  Shared ConvNeXt Backbone
    → Organoid Type Head (脑/肠/肺/...)
    → Per-Type Subtype Heads (各组织的亚型分类)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class MultiOrganoidClassifier(nn.Module):
    """
    输入: (B, 3, H, W) 任意类器官明场图像
    输出: {
        'type_logits': (B, N_tissues),     # 组织类型
        'subtype_logits': {                 # 每组织的亚型
            'brain': (B, 4),
            'intestine': (B, 4),
            'colon': (B, 2),
            ...
        }
    }
    """

    def __init__(self, configs, backbone_name="convnext_tiny"):
        """
        configs: dict like {
            'brain': {'num_classes': 4, 'labels': ['wt2D','A1A-1','B2A-2','TH2-7']},
            'intestine': {'num_classes': 4, 'labels': ['cyst','early','late','spheroid']},
            'colon': {'num_classes': 2, 'labels': ['cystic','solid']},
        }
        """
        super().__init__()
        self.configs = configs
        self.tissue_types = list(configs.keys())

        # ---- Shared Backbone ----
        if "convnext" in backbone_name:
            w = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            full = tv_models.convnext_tiny(weights=w)
            self.backbone = full.features
            self.feature_dim = 768
        elif "efficientnet_b3" in backbone_name:
            w = tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1
            full = tv_models.efficientnet_b3(weights=w)
            self.backbone = full.features
            self.feature_dim = 1536
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # ---- Global Pooling ----
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ---- Tissue Type Head (这是哪个器官的类器官？) ----
        self.tissue_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, len(self.tissue_types)),
        )

        # ---- Per-Tissue Subtype Heads (各组织的亚型分类) ----
        self.subtype_heads = nn.ModuleDict()
        for tissue, cfg in configs.items():
            self.subtype_heads[tissue] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.feature_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, cfg["num_classes"]),
            )

        self._init_weights()

    def _init_weights(self):
        for head in [self.tissue_head] + list(self.subtype_heads.values()):
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward_features(self, x):
        """提取共享特征"""
        f = self.backbone(x)
        f = self.gap(f).flatten(1)
        return f

    def forward(self, x, tissue=None):
        """
        x: 图像 batch
        tissue: 如果已知组织类型（训练时给定），只输出对应的亚型
        """
        f = self.forward_features(x)

        # 组织类型预测
        tissue_logits = self.tissue_head(f)

        # 所有亚型预测
        subtype_logits = {}
        for tissue_name, head in self.subtype_heads.items():
            subtype_logits[tissue_name] = head(f)

        if tissue is not None:
            # 训练模式：返回组织预测 + 指定组织的亚型
            return tissue_logits, subtype_logits[tissue]
        else:
            # 推理模式：返回全部
            return tissue_logits, subtype_logits

    def predict(self, x):
        """推理接口：返回最可能的组织+亚型"""
        tissue_logits, subtype_logits = self.forward(x)
        tissue_idx = tissue_logits.argmax(dim=1)

        results = []
        for i in range(len(x)):
            t_idx = tissue_idx[i].item()
            t_name = self.tissue_types[t_idx]
            s_logits = subtype_logits[t_name][i]
            s_pred = s_logits.argmax().item()
            results.append({
                "tissue": t_name,
                "subtype": self.configs[t_name]["labels"][s_pred],
                "tissue_conf": F.softmax(tissue_logits[i], dim=0).max().item(),
                "subtype_conf": F.softmax(s_logits, dim=0).max().item(),
            })
        return results


# ============================================================
# 预设配置
# ============================================================

PRESET_CONFIGS = {
    "brain": {
        "num_classes": 4,
        "labels": ["wt2D (健康)", "A1A-1 (疾病)", "B2A-2 (疾病)", "TH2-7 (疾病)"],
        "dataset": "OrganoIDNet",
        "num_images": 1407,
    },
    "intestine": {
        "num_classes": 4,
        "labels": ["cyst (囊性早期)", "early_budding (早期出芽)", "late_budding (晚期出芽)", "spheroid (球体)"],
        "dataset": "CLORG-Intestinal",
        "num_images": 23063,
    },
    "colon": {
        "num_classes": 4,
        "labels": ["budding_opaque", "budding_transparent", "nonbudding_opaque", "nonbudding_transparent"],
        "dataset": "CLORG-Colon",
        "num_images": 2477,
    },
    "colorectal_ca": {
        "num_classes": 2,
        "labels": ["cystic (囊性)", "solid (实心)"],
        "dataset": "CRC",
        "num_images": 48000,
    },
}

# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Organoid Classifier — 模型验证")
    print("=" * 60)

    # 只用 brain + intestine 测试（两种组织）
    configs = {k: PRESET_CONFIGS[k] for k in ["brain", "intestine"]}
    model = MultiOrganoidClassifier(configs, backbone_name="convnext_tiny")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    # 模拟输入
    x = torch.randn(4, 3, 512, 512)

    # 测试：已知组织类型 (训练模式)
    tissue_logits, brain_logits = model(x, tissue="brain")
    print(f"\n训练模式 (已知组织=brain):")
    print(f"  Tissue logits: {tissue_logits.shape}")  # (4, 2)
    print(f"  Brain subtype: {brain_logits.shape}")  # (4, 4)

    # 测试：推理模式
    results = model.predict(x)
    print(f"\n推理模式 (自动判别):")
    for i, r in enumerate(results):
        print(f"  Image {i+1}: {r['tissue']} → {r['subtype']} "
              f"(tissue_conf={r['tissue_conf']:.2f}, "
              f"subtype_conf={r['subtype_conf']:.2f})")
