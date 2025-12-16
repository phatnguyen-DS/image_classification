import torch
import onnxruntime as ort
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from pathlib import Path
from src.data.dataset import ISICDataset
from torchvision import transforms

BASE_DIR = Path.cwd().parent.parent
onnx_path = BASE_DIR / "model" / "resnet50_final.onnx"
ort_session = ort.InferenceSession(str(onnx_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

all_test_preds = []
all_test_targets = []
# Target class names
TARGET_CLASSES = sorted(['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC'])

# Chuẩn bị DataLoader cho tập test
TEST_DIR = BASE_DIR / "data" / "processed_test"
IMG_SIZE = 224
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(*stats),
])
test_dataset = ISICDataset(TEST_DIR, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Data Loader với tqdm
test_bar = tqdm(test_loader, desc=f"Epoch [Val] (ONNX)", leave=False)

# Tensor sang Numpy
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

for images, labels in test_bar:
    ort_inputs = {input_name: to_numpy(images)}
    
    ort_outs = ort_session.run(None, ort_inputs)

    outputs = ort_outs[0]
    preds = np.argmax(outputs, axis=1)

    all_test_preds.extend(preds)
    all_test_targets.extend(to_numpy(labels))

precision = precision_score(all_test_targets, all_test_preds, average='weighted', zero_division=0)
recall = recall_score(all_test_targets, all_test_preds, average='weighted', zero_division=0)
f1 = f1_score(all_test_targets, all_test_preds, average='weighted', zero_division=0)

print(f"Precision:  {precision:.4f}   | Recall:    {recall:.4f} | F1: {f1:.4f}")
print("\n--- Classification Report ---")
print(classification_report(all_test_targets, all_test_preds, target_names=TARGET_CLASSES, zero_division=0))
print("-----------------------------")