# Dataset Integration Guide

This document explains how to integrate the 4 public datasets into the project
for training the unified YOLO11s model on the canonical 8-class schema.

---

## 1. Dataset Registry

| Dataset | Source | License | Size | Key Classes |
| :--- | :--- | :--- | :--- | :--- |
| **TACO** | [tacodataset.org](http://tacodataset.org/) / [Zenodo 3354286](https://zenodo.org/records/3354286) | CC BY 4.0 | ~1,500 images, 4,784 annotations | 60 sub-categories → 28 super-categories |
| **Garbage Object Detection** | [huggingface.co/keremberke](https://huggingface.co/datasets/keremberke/garbage-object-detection) | MIT | ~10,464 images | bottle, can, bag, paper, cup, plastic, tissue, trash |
| **RoLID-11K** | [github.com/xq141839/RoLID-11K](https://github.com/xq141839/RoLID-11K) | CC BY 4.0 | 11,000+ dashcam frames | roadside litter (small-object, long-tail) |
| **pLitterStreet** | [Zenodo 8288500](https://zenodo.org/records/8288500) | CC BY 4.0 | 13,000+ images | street-level plastic litter |
| **WADE-AI** | [github.com/letsdoitworld/wade-ai](https://github.com/letsdoitworld/wade-ai) | MIT | Roboflow export | bottle, bag, cup, can, paper, tissue |

---

## 2. Canonical 8-Class Schema

```
0: person
1: bottle
2: cup
3: can
4: bag
5: paper
6: crumpled_tissue
7: other_litter
```

All dataset class names map to these 8 canonical IDs via `scripts/datasets/unified_class_map.py`.

---

## 3. Integration Workflow (Step by Step)

### Step 1: Download & convert annotations
```powershell
python scripts\datasets\prepare_datasets.py --dataset taco --output datasets\taco
python scripts\datasets\prepare_datasets.py --dataset garbage --output datasets\garbage
python scripts\datasets\prepare_datasets.py --dataset rolid --output datasets\rolid
python scripts\datasets\prepare_datasets.py --dataset plitterstreet --output datasets\plitterstreet
python scripts\datasets\prepare_datasets.py --dataset wade --output datasets\wade
```

### Step 2: Auto-label FAILING frames (optional, for crumpled_tissue)
```powershell
# Requires: pip install groundingdino
python scripts\datasets\auto_label_grounding_dino.py --video your_failing_video.mp4 --output labels\ --cues "crumpled tissue paper" "small white litter" "plastic bottle"
```

### Step 3: Merge into YOLO train/val/test structure
Combine the converted labels + your CCTV frames into:
```
datasets/
├── train/
│ ├── images/
│ └── labels/
├── val/
│ ├── images/
│ └── labels/
└── test/
  ├── images/
  └── labels/
```

### Step 4: Generate data.yaml
```powershell
python scripts\train_yolo.py --make-yaml --data datasets\
```

### Step 5: Train
```powershell
# CPU (laptop):
python scripts\train_yolo.py --data datasets\data.yaml --weights yolov8s.pt --epochs 50 --batch 8 --img 640 --device cpu --name litter_yolo11s_v1

# GPU (if available):
python scripts\train_yolo.py --data datasets\data.yaml --weights yolov8s.pt --epochs 50 --batch 16 --img 640 --device 0 --name litter_yolo11s_v1
```

### Step 6: Verify on the FAILING video
```powershell
# Before replacing best.pt, test the new weights on the SAME failing video:
python scripts\run_pipeline.py --source file --video your_failing_video.mp4 --show

# Point the pipeline to the new weights:
python scripts\run_pipeline.py --source file --video your_failing_video.mp4 --show --litter-weights runs\detect\litter_yolo11s_v1\weights\best.pt
```

### Step 7: Commit & push
```powershell
git add -A
git commit -m "add tissue/litter datasets + retrain YOLO11s, fixes littering detection"
git push
```

---

## 4. Important Rules

- **DO NOT replace `best.pt`** until the retrained model measurably outperforms it on the SAME failing video.
- **DO NOT mix TACO/Garbage classes automatically with `crumpled_tissue`** — the crumpled_tissue class must come from YOUR CCTV frames or manually verified labels, not auto-generated from public data.
- **License check**: all 5 datasets are CC BY 4.0 or MIT — safe for academic use with attribution. Verify before any redistribution.
- **Keep best.pt as fallback**: the pipeline already falls back to COCO classes when best.pt is absent; the retrained model only replaces it after verification.
