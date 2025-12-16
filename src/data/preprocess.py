import os
import glob
import shutil
import random
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from pathlib import Path

BASE_DIR = Path.cwd().parent.parent

'''========CONFIG========'''
BASE_DIR = Path.cwd().parent.parent
DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw" / "ISIC 2019 Skin Lesion images for classification"
RAW_TRAIN_DIR = RAW_DATA_DIR / "train"
PROCESSED_TRAIN_DIR = DATA_DIR / "processed"
PROCESSED_VALID_DIR = DATA_DIR / "processed_valid"
PROCESSED_TEST_DIR = DATA_DIR / "processed_test"

IMG_SIZE = (224, 224)
SEED = 42

VALID_CLASSES = ['NV', 'MEL', 'BCC', 'BKL', 'AK', 'SCC', 'VASC', 'DF']
SPLIT_RATIO = {'train': 0.7, 'valid': 0.15, 'test': 0.15}

AUGMENT_STRATEGY = {
    'NV':   0,      
    'MEL':  0.2,    
    'BCC':  0.5,    
    'BKL':  1.0,    
    'AK':   4.0,    
    'SCC':  6.0,    
    'VASC': 13.0,   
    'DF':   14.0    
}

'''========TRANSFORM========'''

class SquarePad:

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return TF.pad(image, padding, 0, 'constant')

# goc
transform_base = transforms.Compose([
    SquarePad(),
    transforms.Resize(IMG_SIZE),
])

# nhe
aug_light = transforms.Compose([
    SquarePad(),
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
])

# trung binh
aug_medium = transforms.Compose([
    SquarePad(),
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])

# nang cao
aug_heavy = transforms.Compose([
    SquarePad(),
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(180),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
])

'''========PROCESSING FUNCTIONS========'''

def save_image(img_pil, path):
    try:
        img_pil.save(path, quality=95)
    except Exception as e:
        print(f"Error saving {path}: {e}")

def process_batch(image_list, output_dir, class_name, is_train=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    count_generated = 0
    
    # --- ẢNH GỐC ---
    for img_path in image_list:
        try:
            img = Image.open(img_path).convert("RGB")
            img_base = transform_base(img)
            fname = os.path.basename(img_path)
            save_image(img_base, os.path.join(output_dir, fname))
        except Exception as e:
            print(f"Skipping corrupt file {img_path}: {e}")

    # --- AUGMENTATION ---
    if is_train:
        factor = AUGMENT_STRATEGY.get(class_name, 0)
        
        if factor > 0:
            num_original = len(image_list)
            num_to_generate = int(num_original * factor)
            
            # Chọn bộ Transform phù hợp
            if factor <= 0.5:
                tf = aug_light
            elif factor <= 4.0:
                tf = aug_medium
            else:
                tf = aug_heavy

            print(f"   [Augment] Generating {num_to_generate} new images (Factor: {factor})...")
            
            sources = random.choices(image_list, k=num_to_generate)
            
            for idx, src_path in enumerate(sources):
                try:
                    img = Image.open(src_path).convert("RGB")
                    img_aug = tf(img)
                    
                    fname_no_ext = os.path.splitext(os.path.basename(src_path))[0]
                    save_name = f"aug_{idx}_{fname_no_ext}.jpg"
                    save_image(img_aug, os.path.join(output_dir, save_name))
                    count_generated += 1
                except Exception:
                    pass
    
    return count_generated

'''========MAIN PIPELINE========'''

def run_preprocess():
    random.seed(SEED)
    
    if not os.path.exists(RAW_TRAIN_DIR):
        print(f"ERROR: Directory not found: {RAW_TRAIN_DIR}")
        return

    stats = []

    # Duyệt qua các class trong VALID_CLASSES
    for class_name in VALID_CLASSES:
        class_src_dir = os.path.join(RAW_TRAIN_DIR, class_name)
        
        if not os.path.isdir(class_src_dir):
            print(f"Warning: Class directory '{class_name}' not found inside raw/train. Skipping.")
            continue
            
        print(f"\nProcessing Class: {class_name}")
        
        # Lấy tất cả ảnh
        all_images = glob.glob(os.path.join(class_src_dir, "*.jpg"))
        
        random.shuffle(all_images)
        total = len(all_images)

        # Chia tách dữ liệu (Split)
        n_train = int(total * SPLIT_RATIO['train'])
        n_valid = int(total * SPLIT_RATIO['valid'])
        
        train_imgs = all_images[:n_train]
        valid_imgs = all_images[n_train : n_train + n_valid]
        test_imgs = all_images[n_train + n_valid:]
        
        dest_train = os.path.join(PROCESSED_TRAIN_DIR, class_name)
        n_aug = process_batch(train_imgs, dest_train, class_name, is_train=True)
        
        dest_valid = os.path.join(PROCESSED_VALID_DIR, class_name)
        process_batch(valid_imgs, dest_valid, class_name, is_train=False)
        
        dest_test = os.path.join(PROCESSED_TEST_DIR, class_name)
        process_batch(test_imgs, dest_test, class_name, is_train=False)
        
        stats.append({
            "Class": class_name,
            "Original": total,
            "Train_Final": len(train_imgs) + n_aug,
            "Valid": len(valid_imgs),
            "Test": len(test_imgs)
        })
    print("Preprocessing Completed Successfully!")

if __name__ == "__main__":
    run_preprocess()