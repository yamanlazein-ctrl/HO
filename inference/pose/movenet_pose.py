"""
MoveNet Pose — 🟢 wrapper.

MoveNet (TensorFlow Lite / TF.js) gives 17 body keypoints per person.
We only consume the subset the associator needs:
    left_wrist, right_wrist, left_shoulder, right_shoulder
and derive ``torso_center`` as the midpoint of the shoulders.

Why MoveNet over a heavier pose model: it is fast on CPU (the demo
target), and the reference project already integrates it. We run it
**lazily** — only on persons that the associator is currently tracking
or considering, not on every detected person every frame. This is the
single biggest CPU saving in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np  # type: ignore

from inference.association.person_object_assoc import Keypoints

# MoveNet 17-keypoint indices (Thunderbird/SinglePose)
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6


@dataclass
class PoseResult:
    person_index: int
    keypoints: Keypoints
    confidence: float


class MovenetPose:
    """
    Wraps a TF MoveNet model. Loads lazily.

    The inference input is a cropped person bbox (not the full frame) to
    keep the model small and fast.
    """

    def __init__(self, model_name: str = "movenet_singlepose_thunder", input_size: int = 256) -> None:
        self.model_name = model_name
        self.input_size = input_size
        self._model = None

    def load(self) -> None:
        import tensorflow as tf  # type: ignore
        # hub or tflite; using tf-hub movenet
        if self.model_name == "movenet_singlepose_thunder":
            self._model = tf.lite.Interpreter  # placeholder; real load below
            # The reference project loads via tflite; we mirror the common path:
            import tensorflow_hub as hub  # type: ignore
            self._model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        else:
            import tensorflow_hub as hub  # type: ignore
            self._model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

    def estimate(self, frame, person_bboxes) -> List[PoseResult]:
        """
        Run pose on each person crop. Returns one PoseResult per person.
        ``person_bboxes`` is a list of (x1,y1,x2,y2).
        """
        if self._model is None:
            raise RuntimeError("MovenetPose.load() must be called first")
        import tensorflow as tf  # type: ignore

        results: List[PoseResult] = []
        for idx, bbox in enumerate(person_bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            inp = tf.image.resize_with_pad(tf.convert_to_tensor(crop)[tf.newaxis, ...], self.input_size, self.input_size)
            inp = tf.cast(inp, dtype=tf.int32)
            outs = self._model.signatures["serving_default"](input=inp)
            kps = outs["output_0"].numpy()[0, 0, :, :]  # (17, 3) — y, x, conf
            results.append(self._to_pose_result(idx, kps, (x1, y1, x2, y2)))
        return results

    @staticmethod
    def _to_pose_result(idx: int, kps: np.ndarray, bbox) -> PoseResult:
        def get(i):
            y, x, c = kps[i]
            if c < 0.2:
                return None
            # map back to frame coords
            x1, y1, x2, y2 = bbox
            h = max(1, y2 - y1)
            w = max(1, x2 - x1)
            return (x1 + float(x) * w, y1 + float(y) * h)

        lw = get(KP_LEFT_WRIST)
        rw = get(KP_RIGHT_WRIST)
        ls = get(KP_LEFT_SHOULDER)
        rs = get(KP_RIGHT_SHOULDER)
        tc = None
        if ls is not None and rs is not None:
            tc = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        kp = Keypoints(left_wrist=lw, right_wrist=rw, torso_center=tc, left_shoulder=ls, right_shoulder=rs)
        avg_conf = float(np.mean(kps[:, 2]))
        return PoseResult(person_index=idx, keypoints=kp, confidence=avg_conf)
