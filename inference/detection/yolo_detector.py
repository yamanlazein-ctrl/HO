"""
YOLO Detector — 🟢 wrapper around ultralytics.

Loads ``best.pt`` (the garbage-detection weights from the reference
project) and runs detection per frame. Returns a list of ``Detection``
dataclasses that the pipeline converts into ``Track`` objects after
ByteTrack assigns IDs.

The reference model's classes (plastic bottle, juice cup, tissue paper,
...) are surfaced via :attr:`classes`. Person detection uses the COCO
``person`` class — we load yolov8n.pt (or a configured COCO model) for
people, and the custom ``best.pt`` for litter objects. Two models is
fine on CPU for a demo; we can also fine-tune a single model later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np  # type: ignore


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    centroid: Tuple[float, float]

    @property
    def is_person(self) -> bool:
        return self.class_name.lower() == "person"


class YoloDetector:
    """
    Wraps two ultralytics YOLO models:
      * person_model : COCO person (yolov8n.pt by default)
      * litter_model : custom best.pt for trash classes

    Both are loaded lazily so importing this module does not require
    ultralytics/torch at import time (the pipeline imports this only on
    the laptop, not in the test sandbox).
    """

    def __init__(
        self,
        litter_weights: str = "inference/detection/weights/best.pt",
        person_weights: str = "yolov8n.pt",
        person_conf: float = 0.4,
        litter_conf: float = 0.35,
        device: str = "cpu",
        fallback_coco_classes: bool = True,
    ) -> None:
        self.litter_weights = litter_weights
        self.person_weights = person_weights
        self.person_conf = person_conf
        self.litter_conf = litter_conf
        self.device = device
        # When the litter model (best.pt) is absent, fall back to emitting
        # non-person COCO classes (bottle, cup, ...) from the person model so
        # the pipeline can still track litter-likely objects. This keeps the
        # demo honest when best.pt is not yet installed.
        self._fallback_coco_classes = fallback_coco_classes
        self._person_model = None
        self._litter_model = None
        self._litter_classes: Optional[List[str]] = None

    def load(self) -> None:
        from ultralytics import YOLO  # type: ignore

        self._person_model = YOLO(self.person_weights)
        if os.path.exists(self.litter_weights):
            self._litter_model = YOLO(self.litter_weights)
            self._litter_classes = list(self._litter_model.names.values())
        else:
            # allow running person-only if litter weights absent
            self._litter_model = None
            self._litter_classes = []

    @property
    def litter_classes(self) -> List[str]:
        return self._litter_classes or []

    def detect(self, frame) -> List[Detection]:
        """Single-frame detection (no tracking). Used for eval/inspection."""
        if self._person_model is None:
            raise RuntimeError("YoloDetector.load() must be called first")
        out: List[Detection] = []

        # people (+ fallback non-person COCO classes when the litter model is absent)
        for r in self._person_model(frame, conf=self.person_conf, device=self.device, verbose=False):
            for box in r.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                name = self._person_model.names[cls_id]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if name.lower() == "person":
                    out.append(Detection("person", conf, (x1, y1, x2, y2), (cx, cy)))
                elif self._litter_model is None and self._fallback_coco_classes:
                    # litter model absent → use non-person COCO classes (bottle, cup,
                    # etc.) as a fallback so the pipeline can still track litter-likely
                    # objects. This is the honest path when best.pt is not installed.
                    out.append(Detection(name, conf, (x1, y1, x2, y2), (cx, cy)))

        # litter (custom model)
        if self._litter_model is not None:
            for r in self._litter_model(frame, conf=self.litter_conf, device=self.device, verbose=False):
                for box in r.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = self._litter_model.names[cls_id]
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    out.append(Detection(name, conf, (x1, y1, x2, y2), (cx, cy)))

        return out

    def track(self, frame, persist: bool = True) -> List["TrackedDetection"]:
        """
        Real ByteTrack tracking via ultralytics model.track(...).

        Returns TrackedDetection objects that carry a STABLE track id
        assigned by ByteTrack across frames (persist=True keeps the
        tracker state between calls). This is the production path —
        the Association/FSM/Voting layers REQUIRE stable ids.

        Persons come from the person model; litter objects come from the
        litter model. Both are tracked independently with their own
        model.track(...) call so their ID spaces never collide.
        """
        if self._person_model is None:
            raise RuntimeError("YoloDetector.load() must be called first")
        out: List[TrackedDetection] = []

        # people (ByteTrack, persist keeps IDs stable across calls)
        for r in self._person_model.track(
            frame, conf=self.person_conf, device=self.device,
            tracker="bytetrack.yaml", persist=persist, verbose=False,
        ):
            out.extend(self._parse_tracked(r, is_person=True, allow_fallback=self._litter_model is None and self._fallback_coco_classes))

        # litter (separate tracker instance via separate model.track calls)
        if self._litter_model is not None:
            for r in self._litter_model.track(
                frame, conf=self.litter_conf, device=self.device,
                tracker="bytetrack.yaml", persist=persist, verbose=False,
            ):
                out.extend(self._parse_tracked(r, is_person=False))

        return out

    @staticmethod
    def _dedup_contained(dets: List["TrackedDetection"], contain_thresh: float = 0.7) -> List["TrackedDetection"]:
        """Suppress partial duplicate detections of the SAME physical object.

        Standard NMS only removes boxes with IoU > ~0.55, so a small box
        fully contained inside a bigger one (e.g. a bottle's cap/neck
        detected separately from the whole bottle) survives and competes in
        tracking. Real-video probing showed these phantoms create ghost
        pairs in the association layer. Rule: process by confidence desc;
        drop any detection whose overlap with an already-kept detection
        covers >= ``contain_thresh`` of ITS OWN area (same person/object
        group). This is standard containment-NMS practice.
        """
        kept: List[TrackedDetection] = []
        for d in sorted(dets, key=lambda t: t.confidence, reverse=True):
            x1, y1, x2, y2 = d.bbox
            area = max(1e-6, (x2 - x1) * (y2 - y1))
            contained = False
            for k in kept:
                if k.is_person != d.is_person:
                    continue
                kx1, ky1, kx2, ky2 = k.bbox
                ix = max(0.0, min(x2, kx2) - max(x1, kx1))
                iy = max(0.0, min(y2, ky2) - max(y1, ky1))
                if (ix * iy) / area >= contain_thresh:
                    contained = True
                    break
            if not contained:
                kept.append(d)
        return kept

    def _parse_tracked(self, r, is_person: bool, allow_fallback: bool = False) -> List["TrackedDetection"]:
        out: List[TrackedDetection] = []
        boxes = r.boxes
        if boxes.id is None:
            return out
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(float, boxes.xyxy[i].tolist())
            conf = float(boxes.conf[i])
            tid = int(boxes.id[i])
            cls_id = int(boxes.cls[i])
            if is_person:
                raw_name = self._person_model.names[cls_id]
            else:
                raw_name = self._litter_model.names[cls_id]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            if is_person and raw_name.lower() == "person":
                out.append(TrackedDetection(
                    track_id=tid, class_name="person", confidence=conf,
                    bbox=(x1, y1, x2, y2), centroid=(cx, cy), is_person=True,
                ))
            elif is_person and allow_fallback:
                # litter model absent → emit non-person COCO classes (bottle, cup, ...)
                # as fallback litter-likely objects, flagged is_person=False so they
                # get namespaced into the object ID space (10000+).
                out.append(TrackedDetection(
                    track_id=tid, class_name=raw_name, confidence=conf,
                    bbox=(x1, y1, x2, y2), centroid=(cx, cy), is_person=False,
                ))
            elif not is_person:
                out.append(TrackedDetection(
                    track_id=tid, class_name=raw_name, confidence=conf,
                    bbox=(x1, y1, x2, y2), centroid=(cx, cy), is_person=False,
                ))
        return self._dedup_contained(out)

    def _model_name_for(self, is_person: bool, cls_id: int) -> str:
        if is_person:
            return "person"
        if self._litter_model is None:
            return f"object_{cls_id}"
        return str(self._litter_model.names.get(cls_id, f"object_{cls_id}"))


@dataclass
class TrackedDetection:
    """A detection with a stable ByteTrack-assigned id."""
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    is_person: bool
