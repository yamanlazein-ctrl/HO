"""
Generate a realistic test video for AI Littering Detection testing without a live camera.
Generates an MP4 where a person holds a bottle, drops it to the ground, and walks away.
"""
import os
import sys
import cv2
import numpy as np

def generate_sample_video(output_path="sample_littering.mp4", duration_sec=6, fps=15):
    print(f"[INFO] Generating test video: {output_path} ({duration_sec}s @ {fps}fps)...")
    
    # Try to load real person image if available from ultralytics assets
    import ultralytics
    ult_dir = os.path.dirname(ultralytics.__file__)
    sample_img_path = os.path.join(ult_dir, "assets", "zidane.jpg")
    
    out_w, out_h = 800, 600
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    if os.path.exists(sample_img_path):
        src_img = cv2.imread(sample_img_path)
        h_s, w_s = src_img.shape[:2]
        # Crop person area
        crop_person = src_img[:, :w_s//2].copy()
        cp_h, cp_w = crop_person.shape[:2]
        scale = 360 / cp_h
        person_scaled = cv2.resize(crop_person, (int(cp_w * scale), 360))
    else:
        # Fallback drawn silhouette
        person_scaled = np.full((360, 140, 3), 120, dtype=np.uint8)
        cv2.circle(person_scaled, (70, 50), 30, (200, 180, 160), -1)
        cv2.rectangle(person_scaled, (40, 80), (100, 240), (160, 100, 60), -1)

    p_h, p_w = person_scaled.shape[:2]
    total_frames = duration_sec * fps
    
    # Simple bottle patch (green bottle)
    bottle_w, bottle_h = 30, 80
    bottle_patch = np.full((bottle_h, bottle_w, 3), 40, dtype=np.uint8)
    bottle_patch[:, :] = (30, 180, 50)  # Green bottle
    cv2.rectangle(bottle_patch, (5, 0), (25, 15), (200, 200, 200), -1)  # Cap

    start_x = 80
    end_x = out_w - p_w - 40

    for i in range(total_frames):
        frame = np.full((out_h, out_w, 3), 35, dtype=np.uint8)
        
        # Ground line indicator
        cv2.line(frame, (0, out_h - 60), (out_w, out_h - 60), (60, 60, 60), 2)
        
        # Person walking right
        px = int(start_x + i * (end_x - start_x) / (total_frames - 1))
        py = out_h - p_h - 60
        
        # Place person
        frame[py:py+p_h, px:px+p_w] = person_scaled
        
        # Bottle animation:
        # 0 to 30% time: HOLDING in hand
        # 30% to 60% time: RELEASE & falling
        # 60% to 100% time: GROUND & stationary while person leaves
        release_frame = int(total_frames * 0.3)
        ground_frame = int(total_frames * 0.6)
        
        if i < release_frame:
            # Holding in right hand
            bx = px + p_w - 10
            by = py + int(p_h * 0.45)
        elif i < ground_frame:
            # Falling downward
            frac = (i - release_frame) / max(1, (ground_frame - release_frame))
            bx = px + p_w - 10
            by = int((py + p_h * 0.45) + frac * (out_h - 60 - (py + p_h * 0.45) - bottle_h))
        else:
            # Stationary on ground at dropped X location
            drop_x = int(start_x + release_frame * (end_x - start_x) / (total_frames - 1)) + p_w - 10
            bx = drop_x
            by = out_h - 60 - bottle_h
            
        bx = max(0, min(bx, out_w - bottle_w))
        by = max(0, min(by, out_h - bottle_h))
        frame[by:by+bottle_h, bx:bx+bottle_w] = bottle_patch
        
        vw.write(frame)
        
    vw.release()
    print(f"[OK] Video created successfully: {output_path}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_littering.mp4"
    generate_sample_video(out)
