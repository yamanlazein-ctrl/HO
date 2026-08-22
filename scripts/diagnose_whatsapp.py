#!/usr/bin/env python3
"""Detailed diagnostics for the WhatsApp real video failure - 379 frames."""
import os, sys, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

VIDEO = r"B:\HO\backend\uploaded_videos\20260821_180002_WhatsApp Video 2026-08-21 at 5.15.03 PM.mp4"

from inference.capture.camera_source import VideoFileSource
from inference.detection.yolo_detector import YoloDetector
from inference.pose.movenet_pose import MovenetPose
from inference.tracking.bytetrack_tracker import BytetrackTracker
from inference.association.person_object_assoc import AssociationConfig, PersonObjectAssociator
from inference.pipeline import InferencePipeline, PipelineConfig
from scripts.run_pipeline import build_tracks_real

def main():
    src = VideoFileSource(VIDEO)
    if not src.open():
        print("FAIL open video")
        return
    print(f"VIDEO: {VIDEO}")
    print(f"total_frames={src.total_frames} fps={src.fps:.2f} duration={src.duration_seconds:.2f} {src.width}x{src.height}")

    detector = YoloDetector()
    detector.load()
    print(f"YOLO litter_classes: {detector.litter_classes}")
    print(f"YOLO person model names: person at 0 = {detector._person_model.names[0]}")
    if detector._litter_model:
        print(f"litter model names: {detector._litter_model.names}")
    else:
        print("litter model MISSING - using COCO fallback")

    movenet = MovenetPose()
    movenet.load()
    print(f"MoveNet loaded_from={movenet.loaded_from}")

    tracker = BytetrackTracker()
    tracker.load()

    cfg = AssociationConfig()
    print(f"AssociationConfig: litter_candidate_classes={cfg.litter_candidate_classes}")
    print(f"  bind_radius={cfg.bind_radius} torso_radius={cfg.torso_radius} min_persistence={cfg.min_persistence}")
    print(f"  frame_height={cfg.frame_height} ground_band_ratio={cfg.ground_band_ratio}")
    # also check pipeline config
    pipe_cfg = PipelineConfig(buffer_seconds=6.0, analysis_fps=30.0, pre_seconds=3.0, post_seconds=3.0)
    pipe_cfg.assoc_config.frame_height = 720  # will be updated per frame? check

    associator = PersonObjectAssociator(cfg)

    # For timing bottleneck
    yolo_times=[]; movenet_times=[]; assoc_times=[]; total_times=[]
    # Per frame logs
    frame_idx=0
    person_ids_seen=set()
    object_ids_seen=set()
    litter_candidate_classes=cfg.litter_candidate_classes

    # To limit output, sample every 10 frames detailed + first 30 frames all
    for pkt in src:
        t_frame_start=time.time()
        # YOLO track
        t0=time.time()
        tracked = detector.track(pkt.frame, persist=True)
        t_yolo=(time.time()-t0)*1000

        # Build tracks
        t1=time.time()
        persons, objects = build_tracks_real(pkt.frame, tracked, movenet, tracker, frame_idx)
        t_movenet=(time.time()-t1)*1000

        # Log raw detections
        if frame_idx < 40 or frame_idx % 20==0:
            print(f"\n--- FRAME {frame_idx} ts={pkt.timestamp:.2f} ---")
            print(f" tracked raw: {len(tracked)}")
            for td in tracked:
                print(f"  raw {td.track_id}:{td.class_name} conf={td.confidence:.2f} bbox={tuple(map(int,td.bbox))} centroid={tuple(map(int,td.centroid))} is_person={td.is_person}")
            print(f" persons Track: {len(persons)}")
            for p in persons:
                kp=p.keypoints
                lw = kp.left_wrist if kp else None
                rw = kp.right_wrist if kp else None
                tc = kp.torso_center if kp else None
                print(f"  P{p.track_id} centroid={tuple(map(int,p.centroid))} bbox={tuple(map(int,p.bbox))} lw={lw} rw={rw} tc={tc}")
            print(f" objects Track: {len(objects)}")
            for o in objects:
                is_cand = any(c in o.class_name.lower() for c in litter_candidate_classes)
                print(f"  O{o.track_id}:{o.class_name} conf? bbox={tuple(map(int,o.bbox))} centroid={tuple(map(int,o.centroid))} is_candidate={is_cand}")
            # Test class gating
            for o in objects:
                is_cand = any(c in o.class_name.lower() for c in litter_candidate_classes)
                print(f"    class_gate O{o.track_id} '{o.class_name}' -> {'ACCEPTED' if is_cand else 'REJECTED BY CLASS GATE'}  accepted={litter_candidate_classes}")

        for p in persons:
            person_ids_seen.add(p.track_id)
        for o in objects:
            object_ids_seen.add((o.track_id, o.class_name))

        # Association
        t2=time.time()
        pair_obs = associator.update(persons, objects, pkt.timestamp)
        t_assoc=(time.time()-t2)*1000

        if frame_idx < 80 or frame_idx % 15==0:
            # For association diagnostics, also check distances manually for first person-object
            if persons and objects:
                p0=persons[0]
                o0=objects[0]
                kp=p0.keypoints
                if kp:
                    d_wrist_vals=[]
                    for w in [kp.left_wrist, kp.right_wrist]:
                        if w:
                            d=math.hypot(w[0]-o0.centroid[0], w[1]-o0.centroid[1])
                            d_wrist_vals.append(d)
                    d_wrist = min(d_wrist_vals) if d_wrist_vals else math.inf
                    d_torso = math.hypot(kp.torso_center[0]-o0.centroid[0], kp.torso_center[1]-o0.centroid[1]) if kp.torso_center else math.inf
                    # Check persistence
                    key=(p0.track_id, o0.track_id)
                    mem=associator._pairs.get(key)
                    hits = len([h for _,h in mem.prox_hits if h]) if mem else 0
                    total_hits = len(mem.prox_hits) if mem else 0
                    est = mem.established if mem else False
                    print(f"  ASSOC DIAG P{p0.track_id}->O{o0.track_id} d_wrist={d_wrist:.1f} d_torso={d_torso:.1f} bind_radius={cfg.bind_radius} held={d_wrist < cfg.bind_radius if d_wrist!=math.inf else False} hits={hits}/{total_hits} need={cfg.min_persistence} est={est} pair_obs={len(pair_obs)}")
                else:
                    print(f"  ASSOC DIAG no keypoints for P{p0.track_id}")

            if pair_obs:
                for po in pair_obs:
                    print(f"  PairObs P{po.person_id} O{po.object_id} {po.object_class} hand_near={po.hand_near_object} moving_down={po.object_moving_down} stationary={po.object_stationary} low={po.object_low} away={po.person_moving_away} regrasp={po.person_re_grasped} d_wrist={po.hand_object_distance}")

        yolo_times.append(t_yolo)
        movenet_times.append(t_movenet)
        assoc_times.append(t_assoc)
        total_times.append((time.time()-t_frame_start)*1000)
        frame_idx+=1
        if frame_idx>=80:
            # For quick test limit to 80 then continue full? but spec wants full 379
            pass
        if frame_idx>=379:
            break

    src.release()
    print("\n=== SUMMARY ===")
    print(f"persons distinct: {person_ids_seen}")
    print(f"objects distinct: {object_ids_seen}")
    print(f"pairs ever: {list(associator._pairs.keys())}")
    for k,m in associator._pairs.items():
        print(f"  pair {k} est={m.established} vanished={m.vanished} hits={list(m.prox_hits)[-5:]} class={m.object_class}")
    if yolo_times:
        print(f"avg yolo {sum(yolo_times)/len(yolo_times):.1f}ms movenet {sum(movenet_times)/len(movenet_times):.1f}ms assoc {sum(assoc_times)/len(assoc_times):.1f}ms total {sum(total_times)/len(total_times):.1f}ms fps {1000/(sum(total_times)/len(total_times)):.1f}")

if __name__=="__main__":
    main()
