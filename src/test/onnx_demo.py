import os
from pathlib import Path
import argparse
import csv
import numpy as np
from PIL import Image
import onnxruntime as ort


TARGET_CLASSES = [
    "AK",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "SCC",
    "VASC",
]

VIETNAMESE_LABELS = {
    "NV": "Nốt ruồi (Nevus)",
    "MEL": "U hắc tố (Melanoma)",
    "BCC": "Ung thư biểu mô tế bào đáy (Basal cell carcinoma)",
    "BKL": "Tăng sừng lành tính (Benign keratosis-like)",
    "AK": "Dày sừng quang hóa (Actinic keratosis)",
    "SCC": "Ung thư biểu mô tế bào vảy (Squamous cell carcinoma)",
    "VASC": "Tổn thương mạch máu (Vascular lesion)",
    "DF": "U sợi da (Dermatofibroma)",
}


def load_session(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def preprocess_image(image_path: str, img_size: int = 224):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


def softmax(x: np.ndarray):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)


def iter_images(folder: Path):
    # If folder has subdirectories as class names, yield (path, truth)
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                p = Path(root) / f
                # truth: parent folder name relative to folder
                rel = p.relative_to(folder)
                truth = rel.parts[0] if len(rel.parts) > 1 else None
                yield p, truth


def main():
    parser = argparse.ArgumentParser(description="ONNX demo inference over a folder of images")
    parser.add_argument("--model", type=str, default="model/resnet50_final.onnx")
    parser.add_argument("--images", type=str, default="data/processed_test")
    parser.add_argument("--out", type=str, default="outputs/onnx_demo_results.csv")
    parser.add_argument("--topk", type=int, default=1)
    args = parser.parse_args()

    model_path = args.model
    images_folder = Path(args.images)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sess = load_session(model_path)
    input_name = sess.get_inputs()[0].name

    rows = []
    total = 0
    correct = 0
    class_counts = {}
    class_correct = {}

    for img_path, truth in iter_images(images_folder):
        total += 1
        try:
            inp = preprocess_image(str(img_path))
            outs = sess.run(None, {input_name: inp})
            logits = outs[0]
            if logits.ndim == 2 and logits.shape[0] == 1:
                logits = logits[0]
            probs = softmax(logits)
            top_idx = int(np.argmax(probs))
            pred_code = TARGET_CLASSES[top_idx] if top_idx < len(TARGET_CLASSES) else str(top_idx)
            pred_vn = VIETNAMESE_LABELS.get(pred_code, pred_code)
            conf = float(probs[top_idx])

            is_correct = False
            if truth is not None:
                # normalize truth case and common labels
                truth_norm = truth.strip()
                if truth_norm == pred_code:
                    is_correct = True

            rows.append({
                "image": str(img_path),
                "truth": truth or "",
                "pred_code": pred_code,
                "pred_vn": pred_vn,
                "confidence": conf,
                "correct": int(is_correct),
            })

            # metrics
            if truth:
                class_counts.setdefault(truth, 0)
                class_counts[truth] += 1
                class_correct.setdefault(truth, 0)
                if is_correct:
                    class_correct[truth] += 1
                    correct += 1

        except Exception as e:
            rows.append({"image": str(img_path), "truth": truth or "", "pred_code": "error", "pred_vn": str(e), "confidence": 0.0, "correct": 0})

    # write CSV
    with open(out_path, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ["image", "truth", "pred_code", "pred_vn", "confidence", "correct"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary
    summary = {
        "total_images": total,
        "total_with_label": sum(class_counts.values()),
        "overall_accuracy": (correct / sum(class_counts.values())) if sum(class_counts.values()) > 0 else None,
        "per_class": {},
    }
    for c, cnt in class_counts.items():
        summary["per_class"][c] = {
            "count": cnt,
            "accuracy": class_correct.get(c, 0) / cnt if cnt > 0 else None,
        }

    summary_path = out_path.parent / "summary.txt"
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write(f"Total images: {total}\n")
        f.write(f"Images with ground-truth label (by folder): {sum(class_counts.values())}\n")
        if summary["overall_accuracy"] is not None:
            f.write(f"Overall accuracy: {summary['overall_accuracy']:.4f}\n")
        else:
            f.write("Overall accuracy: N/A (no labeled subfolders)\n")
        f.write("Per-class:\n")
        for c, v in summary["per_class"].items():
            f.write(f"  {c}: count={v['count']}, accuracy={v['accuracy']:.4f}\n")

    print("Done. Results:")
    print(f"  CSV -> {out_path}")
    print(f"  Summary -> {summary_path}")


if __name__ == '__main__':
    main()
