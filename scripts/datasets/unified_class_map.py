"""
Unified Litter Class Mapping — resolves the class-name mismatch
between the reference best.pt, TACO, Garbage, Roboflow and WADE-AI datasets.

The reference model's best.pt uses:
    ['bottle', 'juice-cup', 'nescafe', 'plate', 'tissue']

TACO uses 60 sub-categories; Garbage Object Detection and Roboflow
use their own class names. This script maps ALL of them to one
canonical 8-class schema so the trainer and the evaluator see consistent labels
regardless of which dataset a sample came from.

Canonical mapping:
    0: person
    1: bottle
    2: cup
    3: can
    4: bag
    5: paper
    6: crumpled_tissue
    7: other_litter
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Canonical 8-class schema (what YOLO will be trained on)
# ---------------------------------------------------------------------------
CANONICAL_CLASSES: Tuple[str, ...] = (
    "person",
    "bottle",
    "cup",
    "can",
    "bag",
    "paper",
    "crumpled_tissue",
    "other_litter",
)

# ---------------------------------------------------------------------------
# Per-dataset → canonical mapping
# ---------------------------------------------------------------------------
# Reference best.pt (ananya868) → canonical
BEST_PT_MAP: Dict[str, int] = {
    "bottle": 1,
    "juice-cup": 2,        # juice cup → cup
    "juice cup": 2,
    "nescafe": 2,        # nescafe cup → cup (beverage cup)
    "plate": 7,           # foam plate → other_litter
    "tissue": 6,          # tissue → crumpled_tissue
    "tissue paper": 6,
}

# TACO (60 sub-categories) → canonical
# Full list at https://tacodataset.org
TACO_MAP: Dict[str, int] = {
    # Bottles & containers
    "Plastic bottle": 1, "Glass bottle": 1, "Bottle": 1,
    "Plastic cap": 7, "Bottle cap": 7,
    # Cups
    "Cup": 2, "Paper cup": 2, "Plastic cup": 2,
    "Foam cup": 2, "Drink can": 2, "Food can": 2,
    # Cans
    "Can": 3, "Aluminum can": 3, "Steel can": 3,
    # Bags
    "Plastic bag": 4, "Paper bag": 4, "Garbage bag": 4,
    "Shopping bag": 4, "Carrier bag": 4,
    # Paper
    "Paper": 5, "Paper sheet": 5, "Paper scrap": 5,
    "Magazine": 5, "Newspaper": 5, "Cardboard": 5,
    # Tissue
    "Tissue paper": 6, "Tissue": 6, "Napkin": 6,
    # Other / unlabeled
    "Other": 7, "Other litter": 7, "Unlabeled litter": 7,
    "Cigarette": 7, "Cigarette butt": 7,
    "Mask": 7, "Glove": 7, "Plastic glove": 7,
    "Plastic container": 7, "Food container": 7,
    "Plastic film": 7, "Plastic wrapper": 7,
    "Plastic straw": 7, "Plastic utensils": 7,
    "Shoe": 7, "Clothing": 7, "Cloth": 7,
    "Tire": 7, "Toy": 7, "Book": 7,
}

# Garbage Object Detection (keremberke) → canonical
GARBAGE_MAP: Dict[str, int] = {
    "trash": 7,
    "garbage": 7,
    "litter": 7,
    "plastic": 1,
    "bottle": 1,
    "can": 3,
    "bag": 4,
    "paper": 5,
    "cup": 2,
    "tissue": 6,
}

# Roboflow trash dataset → canonical
ROBOFLOW_MAP: Dict[str, int] = {
    "bottle": 1, "plastic-bottle": 1, "pet-bottle": 1,
    "cup": 2, "paper-cup": 2,
    "can": 3, "aluminum-can": 3, "drink-can": 3,
    "bag": 4, "plastic-bag": 4, "shopping-bag": 4,
    "paper": 5, "newspaper": 5, "cardboard": 5,
    "tissue": 6, "tissue-paper": 6, "napkin": 6,
    "litter": 7, "trash": 7, "garbage": 7,
}

# WADE-AI → canonical
WADE_MAP: Dict[str, int] = {
    "plastic-bottle": 1, "plastic-bag":4, "plastic-cup":2,
    "aluminum-can":3, "paper-cup":2, "paper-bag":4,
    "tissue-paper":6, "waste":7, "litter":7,
}


def canonical_id(class_name: str, source: str = "auto") -> int:
    """Map any dataset's class name to a canonical id (0-7).

    Args:
        class_name: raw class label from the dataset
        source: one of 'best_pt', 'taco', 'garbage',
              'roboflow', 'wade', or 'auto' (tries all)

    Returns:
        canonical class id (0-7), or -1 if no mapping found.
    """
    cn = class_name.lower().strip()
    maps: List[Dict[str, int]] = []
    if source == "auto":
        maps = [BEST_PT_MAP, TACO_MAP, GARBAGE_MAP, ROBOFLOW_MAP, WADE_MAP]
    elif source == "best_pt":
        maps = [BEST_PT_MAP]
    elif source == "taco":
        maps = [TACO_MAP]
    elif source == "garbage":
        maps = [GARBAGE_MAP]
    elif source == "roboflow":
        maps = [ROBOFLOW_MAP]
    elif source == "wade":
        maps = [WADE_MAP]
    else:
        raise ValueError(f"unknown source: {source}")

    for m in maps:
        # try exact match
        if cn in m:
            return m[cn]
        # try substring match (e.g., 'plastic bottle' contains 'bottle')
        for raw, cid in m.items():
            if raw.lower() in cn:
                return cid
    return -1


def canonical_name(class_id: int) -> str:
    """Return the canonical class name for id 0-7."""
    if 0 <= class_id < len(CANONICAL_CLASSES):
        return CANONICAL_CLASSES[class_id]
    return "unknown"


def write_yolo_labels(
    annotations: List[dict],
    output_dir: Path,
    image_w: int = 1920,
    image_h: int = 1080,
    source: str = "auto",
) -> int:
    """Convert COCO/Roboflow annotations to YOLO format using the unified mapping.

    Each annotation: {image_id, file_name, bbox:[x,y,w,h], category_name, ...}
    Writes one .txt file per image with: class_id cx cy w h (normalised).

    Returns the number of labels written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for ann in annotations:
        cls = canonical_id(ann.get("category_name", ""), source=source)
        if cls < 0:
            continue  # skip unmapped
        x, y, w, h = ann["bbox"]
        # normalise to YOLO format
        cx = (x + w / 2) / image_w
        cy = (y + h / 2) / image_h
        nw = w / image_w
        nh = h / image_h
        img_id = ann.get("image_id", ann.get("file_name", f"img_{count}"))
        txt_path = output_dir / f"{img_id}.txt"
        with open(txt_path, "a") as f:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        count += 1
    return count


if __name__ == "__main__":
    # Print the canonical schema
    print("Canonical 8-class schema:")
    for i, name in enumerate(CANONICAL_CLASSES):
        print(f"  {i}: {name}")
    print()
    print(f"Total canonical classes: {len(CANONICAL_CLASSES)}")
