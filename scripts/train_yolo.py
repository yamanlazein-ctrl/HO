"""
Litter Detection Trainer — trains YOLO11s on the unified 8-class schema.

Combines:
- TACO (real-world litter in context)
- Garbage Object Detection
- WADE-AI / Roboflow litter
- Custom CCTV frames (crumpled_tissue class from your camera)

The trainer is a thin wrapper around ultralytics YOLO training.
It expects labels in YOLO format (from unified_class_map + prepare_datasets + auto_label_grounding_dino).

Usage:
    python scripts/train_yolo.py \\
        --data datasets/ \\
        --weights yolov8s.pt \\
        --epochs 50 \\
        --batch 8 \\
        --img 640 \\
        --device cpu \\
        --name litter_yolo11s_v1

This does NOT replace best.pt unless the new model
measurably outperforms it on the SAME failing video.
Keep best.pt as fallback until the retrained model is verified.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def train_yolo(
    data_yaml: str,
    weights: str = "yolov8s.pt",
    epochs: int = 50,
    batch: int = 8,
    img: int = 640,
    device: str = "cpu",
    name: str = "litter_yolo11s_v1",
) -> bool:
    """Train YOLO11s on the unified dataset.

    Args:
        data_yaml: path to a data.yaml in ultralytics format (train/val/test paths + nc: 8 + names)
        weights: starting weights (yolov8s.pt for speed, yolov8m.pt for accuracy)
        epochs: training epochs (30-50 is enough for fine-tuning on a small dataset)
        batch: batch size (2-8 for CPU, 16+ for GPU)
        img: input image size (640 default; 1280 for small objects)
        device: 'cpu' or '0' (GPU index)
        name: run name for the trained weights

    Returns:
        True if training completed and weights were saved.

    The actual training is delegated to ultralytics. This wrapper
    sets up the data.yaml with the unified 8-class names and runs:
        yolo train model=yolov8s.pt data=data_yaml epochs=50 ...
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    print(f"[INFO] Starting YOLO training: {epochs} epochs, batch={batch}, img={img}, device={device}")
    print(f"  Weights: {weights}")
    print(f"  Data: {data_yaml}")
    print(f"  Classes: 8 (person, bottle, cup, can, bag, paper, crumpled_tissue, other_litter)")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img,
        device=device,
        name=name,
    )
    weights_path = f"runs/detect/{name}/weights/best.pt"
    if os.path.exists(weights_path):
        print(f"[OK] Training complete. Weights saved to: {weights_path}")
        print(f"  To use in the pipeline, point YoloDetector(litter_weights='{weights_path}')")
        return True
    print(f"[ERROR] Training did not produce weights at {weights_path}")
    return False


def make_data_yaml(data_dir: str, output: str = "datasets/data.yaml") -> str:
    """Generate the ultralytics data.yaml with the unified 8-class names."""
    from scripts.datasets.unified_class_map import CANONICAL_CLASSES

    data_dir = Path(data_dir)
    yaml_path = Path(output)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # ultralytics data.yaml format
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CANONICAL_CLASSES))
    content = f"""# Unified 8-class litter detection dataset
path: {data_dir.resolve()}
train: {data_dir}/train/images
val: {data_dir}/val/images
test: {data_dir}/test/images

nc: {len(CANONICAL_CLASSES)}
names:
{names_block}
"""
    yaml_path.write_text(content)
    return str(yaml_path)


def main():
    ap = argparse.ArgumentParser(description="Train YOLO11s on unified litter dataset")
    ap.add_argument("--data", default="datasets", help="dataset root directory")
    ap.add_argument("--weights", default="yolov8s.pt", help="starting weights")
    ap.add_argument("--epochs", type=int, default=50, help="training epochs")
    ap.add_argument("--batch", type=int, default=8, help="batch size")
    ap.add_argument("--img", type=int, default=640, help="input image size")
    ap.add_argument("--device", default="cpu", help="cpu or GPU index")
    ap.add_argument("--name", default="litter_yolo11s_v1", help="run name")
    ap.add_argument("--make-yaml", action="store_true", help="generate data.yaml before training")
    args = ap.parse_args()

    data_yaml = args.data + "/data.yaml"
    if args.make_yaml:
        p = make_data_yaml(args.data, data_yaml)
        print(f"[OK] data.yaml written to {p}")

    ok = train_yolo(
        data_yaml=data_yaml,
        weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        img=args.img,
        device=args.device,
        name=args.name,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
