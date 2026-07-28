"""
src/model/ocunet.py
--------------------
OcuNet — EfficientNet-B0 backbone + custom multi-label classifier head.
Designed for RTX 3050 4GB VRAM with mixed precision training.
"""

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 46


class OcuNet(nn.Module):
    """
    EfficientNet-B0 pretrained backbone with a custom classifier head.
    Outputs 46-dimensional sigmoid vector for multi-label classification.
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.4, pretrained: bool = True):
        super(OcuNet, self).__init__()

        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Keep everything except the final classifier
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        # EfficientNet-B0 outputs 1280 features after avgpool
        in_features = backbone.classifier[1].in_features

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
            # No sigmoid here — BCEWithLogitsLoss handles it during training
            # At inference we apply sigmoid manually
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def predict(self, x, threshold: float = 0.5):
        """Inference — returns binary predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.sigmoid(logits)
            preds  = (probs >= threshold).float()
        return preds, probs


def build_model(config: dict) -> OcuNet:
    model_cfg = config["model"]
    model = OcuNet(
        num_classes=NUM_CLASSES,
        dropout=model_cfg["dropout"],
        pretrained=model_cfg["pretrained"]
    )
    return model


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    model = build_model(config)
    print(model)

    # Verify output shape
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    print(f"\nInput  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")   # Should be (2, 46)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params     : {total_params:,}")
    print(f"Trainable params : {train_params:,}")