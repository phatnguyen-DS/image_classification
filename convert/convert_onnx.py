import torch
import torch.nn as nn
import torchvision.models as models
import onnxruntime as ort
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path.cwd().parent
    
PT_PATH = BASE_DIR / "model" / "best_model.pt"
ONNX_PATH = BASE_DIR / "model" / "resnet50_final.onnx"

DEVICE = torch.device('cpu')

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(2048, 1024),
    nn.LeakyReLU(negative_slope=0.01),
    nn.Dropout(p=0.4),
    nn.Linear(1024, 8),
)

try:
    checkpoint = torch.load(PT_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
except Exception as e:
    print(f"{e}")
    exit()

model.eval()

dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

try:
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=12,          
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"{ONNX_PATH}")
except Exception as e:
    print(f"{e}")
    exit()

pt_out = model(dummy_input).detach().cpu().numpy()

try:
    ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    onnx_out = ort_session.run(None, {'input': dummy_input.numpy()})[0]

    diff = np.max(np.abs(pt_out - onnx_out))

    if diff < 1e-4:
        print("\n ONNX thành công.")
    else:
        print("\nsai số nhỏ.")
except Exception as e:
    print(f"{e}")