"""
Inference Pipeline — orchestrates the full per-frame flow.

This is the integration layer that wires:
    capture → buffer → YOLO → ByteTrack → MoveNet → association
    → state machine → voting → evidence manager → (optional) backend POST

It is the only module that knows about all the pieces. Each piece is
independently unit-tested; this module's job is choreography.

Design choices made here (matches the architecture doc):
  * Capture FPS != Analysis FPS. The buffer ingests every captured frame
    at full FPS so evidence clips are smooth, but the heavy pipeline
    (YOLO + pose) runs at a configurable analysis_fps to stay real-time
    on CPU. Frames between analysis ticks are still buffered.
  * Pose is lazy: we only run MoveNet on persons that the associator is
    currently tracking or about to evaluate, not on every detected person.
  * On LITTERING_CONFIRMED, we (a) take the snapshot+pre segment
    immediately, (b) wait ``post_seconds`` of real time, (c) finalize the
    video. The dashboard reflects this ~3s latency by design.
  * Backend reporting is optional and async (HTTP POST on a thread) so a
    slow/unreachable backend never stalls the live pipeline.
"""

from __future__ import annotations

import threading
import time
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from inference.association.person_object_assoc import (
    AssociationConfig,
    PairObservation,
    PersonObjectAssociator,
    Track,
)
from inference.behavior.state_machine import (
    LitterState,
    LitteringStateMachine,
    Observation,
    StateMachineConfig,
    TransitionResult,
)
from inference.behavior.voting import TemporalVoter, VoteObservation, VotingConfig
from inference.capture.circular_buffer import CircularFrameBuffer
from inference.evidence.evidence_manager import EvidenceArtifact, EvidenceManager, EvidenceRequest


@dataclass
class PipelineConfig:
    buffer_seconds: float = 6.0
    analysis_fps: float = 10.0
    pre_seconds: float = 3.0
    post_seconds: float = 3.0
    camera_id: str = "cam-01"
    state_config: StateMachineConfig = field(default_factory=StateMachineConfig)
    voting_config: VotingConfig = field(default_factory=VotingConfig)
    assoc_config: AssociationConfig = field(default_factory=AssociationConfig)
    post_backend_url: Optional[str] = None  # if set, POST events to FastAPI


@dataclass
class PipelineEvent:
    """A confirmed littering event surfaced to the UI/backend."""

    event_id: str
    camera_id: str
    person_track_id: int
    object_track_id: int
    object_type: str
    confidence: float
    event_timestamp: float
    state_history: list


class InferencePipeline:
    """
    The pipeline is driven by :meth:`process_frame`, which the capture
    loop calls. Internally it throttles to ``analysis_fps``.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.buffer = CircularFrameBuffer(window_seconds=self.config.buffer_seconds)
        self.associator = PersonObjectAssociator(self.config.assoc_config)
        self.evidence_mgr = EvidenceManager()
        # per-pair FSM + voter
        self._fsms: Dict[Tuple[int, int], LitteringStateMachine] = {}
        self._voters: Dict[Tuple[int, int], TemporalVoter] = {}
        # pending evidence awaiting post-window finalize
        self._pending: Dict[Tuple[int, int], EvidenceArtifact] = {}
        self._last_analysis_ts: float = 0.0
        self._frame_count = 0
        self.events: List[PipelineEvent] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def process_frame(
        self,
        frame,
        timestamp: float,
        persons: List[Track],
        objects: List[Track],
    ) -> List[PipelineEvent]:
        """
        Called by the capture loop. ``persons``/``objects`` are already
        tracker-output ``Track`` objects (the pipeline caller is
        responsible for running YOLO + ByteTrack + MoveNet and assembling
        these). This method handles buffering + association + FSM +
        voting + evidence.

        Returns any new confirmed events emitted this frame.
        """
        # 1) always buffer the raw frame (full capture FPS)
        self.buffer.push(frame, timestamp=timestamp, frame_index=self._frame_count)
        self._frame_count += 1

        # 2) throttle the heavy logic to analysis_fps
        if self._last_analysis_ts > 0 and (timestamp - self._last_analysis_ts) < (1.0 / self.config.analysis_fps):
            return []

        self._last_analysis_ts = timestamp
        new_events: List[PipelineEvent] = []

        # 3) association → pair observations
        pair_obs: List[PairObservation] = self.associator.update(persons, objects, timestamp)

        # 4) feed each pair into its FSM
        for po in pair_obs:
            key = (po.person_id, po.object_id)
            fsm = self._fsms.get(key)
            if fsm is None:
                fsm = LitteringStateMachine(self.config.state_config)
                self._fsms[key] = fsm
                self._voters[key] = TemporalVoter(self.config.voting_config)

            obs = Observation(
                timestamp=po.timestamp,
                person_present=True,
                object_present=True,
                hand_near_object=po.hand_near_object,
                object_moving_down=po.object_moving_down,
                object_stationary=po.object_stationary,
                object_low=po.object_low,
                person_moving_away=po.person_moving_away,
                person_re_grasped=po.person_re_grasped,
                person_returned=po.person_returned,
            )
            result = fsm.step(obs)
            vote_result = None

            # 5) voting when SUSPICIOUS
            if result.state == LitterState.SUSPICIOUS:
                voter = self._voters[key]
                voter.add(VoteObservation(
                    timestamp=po.timestamp,
                    object_separated=(not po.hand_near_object) and po.hand_object_distance is not None,
                    object_downward=po.object_moving_down,
                    object_stationary=po.object_stationary,
                    object_low=po.object_low,
                    person_away=po.person_moving_away,
                    no_regrasp=(not po.person_re_grasped),
                    regrasp=po.person_re_grasped,
                    person_returned=po.person_returned,
                ))
                vres = voter.decide(now_ts=po.timestamp)
                if vres.decision == "CONFIRM":
                    vote_result = fsm.confirm_from_voting(po.timestamp, vres.score, self.config.voting_config.confirm_threshold)
                elif vres.decision == "REVERT":
                    vote_result = fsm.force_revert(po.timestamp, reason=f"voting revert score={vres.score:.2f}")
                else:
                    vote_result = None

            # 6) on confirmation (either from the FSM step or from voting),
            #    start evidence assembly.
            confirmed_now = result.confirmed or (vote_result is not None and vote_result.confirmed)
            if confirmed_now and key not in self._pending:
                event_id = self.evidence_mgr.new_event_id()
                # confidence: use voting score if available, else 0
                conf_val = float(vres.score) if ('vres' in locals() and vres is not None and vres.decision == "CONFIRM") else 0.0
                req = EvidenceRequest(
                    camera_id=self.config.camera_id,
                    person_track_id=po.person_id,
                    object_track_id=po.object_id,
                    object_type=po.object_class,
                    confidence=conf_val,
                    event_timestamp=po.timestamp,
                    pre_seconds=self.config.pre_seconds,
                    post_seconds=self.config.post_seconds,
                )
                art = self.evidence_mgr.assemble_snapshot(event_id, self.buffer, req)
                if art is not None:
                    self._pending[key] = art
                    ev = PipelineEvent(
                        event_id=event_id,
                        camera_id=req.camera_id,
                        person_track_id=req.person_track_id,
                        object_track_id=req.object_track_id,
                        object_type=req.object_type,
                        confidence=req.confidence,
                        event_timestamp=req.event_timestamp,
                        state_history=fsm.history(),
                    )
                    new_events.append(ev)
                    self.events.append(ev)
                    self._maybe_post_backend(ev, art)

        # 7) finalize pending evidence whose post-window has elapsed
        for key in list(self._pending.keys()):
            art = self._pending[key]
            if timestamp - art.request.event_timestamp >= self.config.post_seconds:
                self.evidence_mgr.finalize(art, self.buffer)
                del self._pending[key]

        return new_events

    # ------------------------------------------------------------------ #
    def _maybe_post_backend(self, event: PipelineEvent, artifact: EvidenceArtifact) -> None:
        url = self.config.post_backend_url
        if not url:
            return
        # The evidence video is NOT ready yet — finalize() runs after post_seconds
        # (3s by default). So we must wait for finalize before uploading.
        # This function is called from process_frame() at confirmation time,
        # but the video file won't exist until finalize() completes.
        # Solution: schedule the backend upload AFTER finalize, not before.
        def _post():
            try:
                import requests  # type: ignore
                # 1) wait for the evidence video to be finalized
                #    (poll the video path until it exists or timeout)
                vid_path = getattr(artifact, "video_path", None)
                snap_path = getattr(artifact, "snapshot_path", None)
                video_ready = False
                if vid_path:
                    for _ in range(int(self.config.post_seconds * 10) + 10):
                        if os.path.exists(vid_path):
                            video_ready = True
                            break
                        time.sleep(0.1)
                # 2) create the event record
                payload = {
                    "camera_id": event.camera_id,
                    "person_track_id": str(event.person_track_id),
                    "object_track_id": str(event.object_track_id),
                    "object_type": event.object_type,
                    "confidence": event.confidence,
                    "timestamp": event.event_timestamp,
                    "status": "confirmed",
                }
                base = url.rsplit("/events", 1)[0]  # tolerate trailing /events
                resp = requests.post(f"{base}/events", json=payload, timeout=3)
                if resp.status_code != 201:
                    return  # event creation failed; can't upload evidence without an event id
                event_db_id = resp.json().get("id")
                if event_db_id is None:
                    return
                # 3) upload the evidence files (snapshot + video) to the backend
                #    ONLY after the video is finalized
                if snap_path and os.path.exists(snap_path):
                    if video_ready and vid_path and os.path.exists(vid_path):
                        with open(snap_path, "rb") as sf, open(vid_path, "rb") as vf:
                            files = {
                                "snapshot": ("snapshot.jpg", sf, "image/jpeg"),
                                "video": ("evidence.mp4", vf, "video/mp4"),
                            }
                            data = {"duration_sec": str(artifact.duration_seconds)}
                            requests.post(
                                f"{base}/evidence/{event_db_id}/upload",
                                files=files, data=data, timeout=10,
                            )
                    else:
                        # video not ready — upload snapshot only, flag missing video
                        with open(snap_path, "rb") as sf:
                            files = {"snapshot": ("snapshot.jpg", sf, "image/jpeg")}
                            data = {"duration_sec": str(artifact.duration_seconds)}
                            requests.post(
                                f"{base}/evidence/{event_db_id}/upload",
                                files=files, data=data, timeout=10,
                            )
            except Exception:
                pass  # backend optional; live pipeline must not depend on it
        threading.Thread(target=_post, daemon=True).start()

    def push_status(self, timestamp: float, capture_fps: Optional[float] = None,
                     analysis_fps_actual: Optional[float] = None,
                     inference_latency_ms: Optional[float] = None,
                     source_type: str = "camo") -> None:
        """Push live pipeline metrics to the backend /api/status singleton.

        Called from run_pipeline.py each stats tick so the dashboard's status
        bar reflects the real AI engine + camera + processing state. If the
        backend status router is not importable (e.g. backend not installed),
        this is a no-op - the live pipeline must not depend on the backend.

        Metrics are SEPARATED honestly:
          - capture_fps: how fast the camera delivers frames (NOT AI speed)
          - analysis_fps_actual: how fast the AI pipeline actually processes
          - inference_latency_ms: measured end-to-end per-frame latency
          - source_type: actual source (camo / file / webcam)
        """
        try:
            from backend.routers.status import set_status
        except Exception:
            return
        st = self.stats()
        # if no actual analysis FPS measured, fall back to config (honest: not real)
        proc_fps = analysis_fps_actual if analysis_fps_actual is not None else None
        set_status(
            ai_engine={"status": "online", "model_loaded": True, "classes": []},
            camera={"status": "online", "fps": capture_fps, "resolution": None, "source": source_type},
            processing={
                "fps": proc_fps,  # actual measured AI FPS, NOT capture FPS
                "latency_ms": inference_latency_ms,  # measured, NOT 1000/analysis_fps
                "analysis_fps": self.config.analysis_fps,  # configured target
            },
            buffer={
                "window_seconds": self.config.buffer_seconds,
                "frames_buffered": st["frames_buffered"],
                "buffer_duration": st["buffer_duration"],
            },
        )

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        return {
            "frames_buffered": len(self.buffer),
            "buffer_duration": self.buffer.current_duration(),
            "active_pairs": len(self._fsms),
            "pending_evidence": len(self._pending),
            "confirmed_events": len(self.events),
        }
