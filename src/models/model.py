import torch.nn as nn
from torchvision import models

# ==========================================
# MODEL
# ==========================================

# model update weight full
def get_transfer_model():
  model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

  for layer in model.parameters():
    layer.requires_grad =False
  model.fc = nn.Sequential(
      nn.Dropout(p=0.4),
      nn.Linear(in_features=2048, out_features=1024),
      nn.LeakyReLU(),
      nn.Dropout(p=0.4),
      nn.Linear(in_features=1024, out_features=8),
  )

  return model

# freeze weight
def unfreeze_layers(model):
    for name, child in model.named_children():
        if name in ['layer3','layer4']:
            for layer in child.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True