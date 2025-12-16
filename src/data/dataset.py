import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob

# ==========================================
# DATASET
# ==========================================

class ISICDataset(Dataset):
    def __init__(self, data_path, transforms=None, target_classes=None):
        self.data_path = data_path
        self.transforms = transforms
        self.target_classes = target_classes
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.target_classes)}

        self.image_paths = []
        self.labels = []

        for class_name in self.target_classes:
            class_path = os.path.join(data_path, class_name)

            paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
                paths.extend(glob.glob(os.path.join(class_path, ext)))

            class_idx = self.class_to_idx[class_name]
            self.image_paths.extend(paths)
            self.labels.extend([class_idx] * len(paths))

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transforms:
            image = self.transforms(image)
        return image, self.labels[idx]

if __name__ == "__main__":
  pass