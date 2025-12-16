import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report
from tqdm import tqdm
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from src.models.loss import FocalLoss
from src.data.dataset import ISICDataset
from src.models.model import get_transfer_model, unfreeze_layers
from pathlib import Path

BASE_DIR = Path.cwd().parent.parent

# folder
TRAIN_DIR = BASE_DIR / "data" / "processed"
VALID_DIR = BASE_DIR / "data" / "processed_valid"
SAVE_PATH = BASE_DIR / "model" / "best_model.pt"

# class
TARGET_CLASSES = sorted([
    "AK",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "SCC",
    "VASC"
])

BATCH_SIZE = 16
EPOCHS = 52
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# dinh nghia transforms cho tap train va tap test
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),

    transforms.ToTensor(),
    transforms.Normalize(*stats),
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(*stats),
])


# Khoi tao data va dataloader
train_dataset = ISICDataset(TRAIN_DIR, transforms=train_transform, target_classes=TARGET_CLASSES)
valid_dataset = ISICDataset(VALID_DIR, transforms=valid_transform, target_classes=TARGET_CLASSES)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

# Tính Class Weights
class_counts = Counter(train_dataset.labels)
weights = [len(train_dataset) / (len(TARGET_CLASSES) * class_counts[i]) for i in range(len(TARGET_CLASSES))]
class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)

# Model
model = get_transfer_model()
model = model.to(DEVICE)

# loss and optimize
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

writer = SummaryWriter('runs/isic_experiment')

# --- 3. Training Loop ---
best_loss = float('inf')

print("===== START =====")

for epoch in range(EPOCHS):
    if epoch == 8:
        print("\n ===== UNFREEZE LAYER 4 =====")
        unfreeze_layers(model)
        optimizer = optim.AdamW([
            {'params': model.layer3.parameters(), 'lr': 1e-5},
            {'params': model.layer4.parameters(), 'lr': 1e-5},
            {'params': model.fc.parameters(), 'lr': 1e-4}
        ], weight_decay=1e-4)
    # --- TRAINING PHASE ---
    model.train()
    train_loss = 0.0
    all_train_preds = []
    all_train_targets = []

    # Thanh progress bar cho train
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

    for images, labels in train_bar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Lưu dự đoán
        _, preds = torch.max(outputs, 1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_targets.extend(labels.cpu().numpy())

        train_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    # --- VALIDATION ---
    model.eval()
    test_loss = 0.0
    all_test_preds = []
    all_test_targets = []

    test_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)

    with torch.no_grad():
        for images, labels in test_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            all_test_preds.extend(preds.cpu().numpy())
            all_test_targets.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(valid_loader)

    # tinh metrics
    precision = precision_score(all_test_targets, all_test_preds, average='weighted', zero_division=0)
    recall = recall_score(all_test_targets, all_test_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_test_targets, all_test_preds, average='weighted', zero_division=0)

    # --- Ghi Log TensorBoard ---
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Test', avg_test_loss, epoch)
    writer.add_scalar('Metrics/Precision', precision, epoch)
    writer.add_scalar('Metrics/Recall', recall, epoch)
    writer.add_scalar('Metrics/F1', f1, epoch)

    # --- In kết quả ra màn hình ---
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
    print(f"  Precision:  {precision:.4f}   | Recall:    {recall:.4f}")

    # --- Classification Report (Mỗi 2 epoch) ---
    if (epoch + 1) % 2 == 0:
        print("\n--- Classification Report ---")
        print(classification_report(all_test_targets, all_test_preds, target_names=TARGET_CLASSES, zero_division=0))
        print("-----------------------------")

    # --- Lưu Model Tốt Nhất ---
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  [SAVE] Đã lưu model tốt nhất (Loss giảm xuống {best_loss:.4f}) tại: {SAVE_PATH}")
writer.close()

'''========================================= part 2========================================='''

# Model
checkpoint_path = SAVE_PATH
model = get_transfer_model()
model = model.to(DEVICE)
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

try:
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
except Exception as e:
    print(f"loi {e}")

unfreeze_layers(model)

# loss and optimize
criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)


writer = SummaryWriter('runs/isic_experiment')
EPOCHS = 50
# --- 3. Training Loop ---
best_loss = float('inf')

print("===== START =====")

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    train_loss = 0.0
    all_train_preds = []
    all_train_targets = []

    # Thanh progress bar cho train
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

    for images, labels in train_bar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Lưu dự đoán
        _, preds = torch.max(outputs, 1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_targets.extend(labels.cpu().numpy())

        train_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    # --- VALIDATION ---
    model.eval()
    test_loss = 0.0
    all_test_preds = []
    all_test_targets = []

    test_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)

    with torch.no_grad():
        for images, labels in test_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            all_test_preds.extend(preds.cpu().numpy())
            all_test_targets.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(valid_loader)

    # tinh metrics
    precision = precision_score(all_test_targets, all_test_preds, average='weighted', zero_division=0)
    recall = recall_score(all_test_targets, all_test_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_test_targets, all_test_preds, average='weighted', zero_division=0)

    # --- Ghi Log TensorBoard ---
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Test', avg_test_loss, epoch)
    writer.add_scalar('Metrics/Precision', precision, epoch)
    writer.add_scalar('Metrics/Recall', recall, epoch)
    writer.add_scalar('Metrics/F1', f1, epoch)

    # --- In kết quả ra màn hình ---
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
    print(f"  Precision:  {precision:.4f}   | Recall:    {recall:.4f}")

    # --- Classification Report (Mỗi 2 epoch) ---
    if (epoch + 1) % 2 == 0:
        print("\n--- Classification Report ---")
        print(classification_report(all_test_targets, all_test_preds, target_names=TARGET_CLASSES, zero_division=0))
        print("-----------------------------")

    # --- Lưu Model Tốt Nhất ---
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  [SAVE] Đã lưu model tốt nhất (Loss giảm xuống {best_loss:.4f}) tại: {SAVE_PATH}")

print("\n XONG")
writer.close()\

