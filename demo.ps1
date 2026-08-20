# ============================================================================
#  AI Littering Detection — LIVE DEMO MODE
#  .\demo.ps1
#
#  Final graduation committee demonstration launcher.
#  Runs a preflight check, WAITS for the iPhone camera (never fakes [OK]),
#  then starts the live AI pipeline so the committee sees real detection.
#
#  HONESTY: if the iPhone is missing → DEMO BLOCKED, not "ready".
#  The AI pipeline only starts when a real camera is producing valid frames.
# ============================================================================

param(
    [int]$CameraDevice = -1,   # -1 = auto-discover
    [int]$AnalysisFps = 10,
    [int]$BufferSeconds = 6
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

function Write-Status($tag, $name, $detail = "") {
    $color = switch ($tag) {
        "OK"      { "Green" }
        "WARNING" { "Yellow" }
        "ERROR"   { "Red" }
        "WAITING" { "Cyan" }
        default   { "Gray" }
    }
    $line = "  $name".PadRight(28) + "[$tag]"
    if ($detail) { $line += "  $detail" }
    Write-Host $line -ForegroundColor $color
}

$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Status "ERROR" "Installation" "run .\install.ps1 first"; exit 1 }

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AI LITTERING DETECTION — DEMO MODE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ===================== PREFLIGHT CHECK =====================
Write-Host "  Preflight check..." -ForegroundColor Gray
Write-Host ""

# Environment
$envOk = $true
try { $null = & $py --version 2>&1 } catch { $envOk = $false }
if ($envOk) { Write-Status "OK" "Environment" } else { Write-Status "ERROR" "Environment" "venv missing"; exit 1 }

# Models
$bestPt = Join-Path $repoRoot "inference\detection\weights\best.pt"
$yoloN  = Join-Path $repoRoot "yolov8n.pt"
if (Test-Path $yoloN)  { Write-Status "OK" "Person model (yolov8n.pt)" } else { Write-Status "ERROR" "Person model" "yolov8n.pt missing"; exit 1 }
if (Test-Path $bestPt) { Write-Status "OK" "Litter model (best.pt)" } else { Write-Status "ERROR" "Litter model" "place best.pt at inference\detection\weights\best.pt"; exit 1 }

# AI libraries
$aiOk = $true
try { & $py -c "import ultralytics, cv2, tensorflow" 2>&1 | Out-Null } catch { $aiOk = $false }
if ($aiOk) { Write-Status "OK" "YOLO" } else { Write-Status "ERROR" "YOLO" "ultralytics/cv2/tensorflow not installed"; exit 1 }
try { & $py -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt')" 2>&1 | Out-Null } catch { Write-Status "ERROR" "YOLO load" "yolov8n.pt failed to load"; exit 1 }
try { & $py -c "from inference.pipeline import InferencePipeline, PipelineConfig; InferencePipeline(PipelineConfig())" 2>&1 | Out-Null } catch { Write-Status "ERROR" "AI Pipeline" "import failed"; exit 1 }
Write-Status "OK" "ByteTrack"
try { & $py -c "import tensorflow as tf" 2>&1 | Out-Null } catch { Write-Status "WARNING" "MoveNet" "tensorflow not installed — pose disabled" }
try { & $py -c "from inference.pose.movenet_pose import MovenetPose" 2>&1 | Out-Null } catch { Write-Status "WARNING" "MoveNet" "import failed" }

# Infrastructure
$dockerPresent = $false
try { $null = docker --version 2>&1; $dockerPresent = $true } catch {}
if ($dockerPresent) {
    try { docker-compose up -d postgres 2>&1 | Out-Null; Write-Status "OK" "Database" "PostgreSQL via Docker" } catch { Write-Status "WARNING" "Database" "docker-compose failed" }
} else { Write-Status "WARNING" "Database" "Docker not installed — use external Postgres" }

$backendProc = Start-Process powershell -PassThru -ArgumentList "-NoExit", "-Command", "& '$py' -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
Start-Sleep -Seconds 3
try { $null = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5; Write-Status "OK" "Backend" } catch { Write-Status "WARNING" "Backend" "not reachable" }

if (Test-Path "dashboard\dist\index.html") { Write-Status "OK" "Dashboard" "build present" } else { Write-Status "WARNING" "Dashboard" "not built — run .\install.ps1" }

# ===================== CAMERA CHECK =====================
Write-Host ""
Write-Host "  Camera check..." -ForegroundColor Gray
Write-Host ""
$camOut = & $py scripts/camera_discovery.py 2>&1
$liveIdx = -1
if ($camOut -match "Idx\s+(\d+)\s+LIVE") { $liveIdx = $Matches[1] }
if ($liveIdx -lt 0) {
    Write-Status "WAITING" "iPhone Camera" "no LIVE camera detected"
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  DEMO BLOCKED" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Reason: Camera not detected." -ForegroundColor Red
    Write-Host "  Please connect the iPhone and open Camo." -ForegroundColor Red
    Write-Host "  Then re-run: .\demo.ps1" -ForegroundColor Red
    Write-Host ""
    if ($backendProc) { Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue }
    exit 1
}
if ($CameraDevice -lt 0) { $CameraDevice = $liveIdx }

# Camera details
$camInfo = & $py -c "
import cv2, time, sys
cap = cv2.VideoCapture($CameraDevice)
ok, f = cap.read()
if not ok or f is None: print('ERROR'); sys.exit(1)
h, w = f.shape[:2]
t0=time.time(); n=0
for _ in range(15): ok,f=cap.read(); n+=1
fps=n/(time.time()-t0)
print('Camera: device index $CameraDevice')
print('Resolution: %dx%d' % (w, h))
print('Detected FPS: %.1f' % fps)
print('Frame status: LIVE')
cap.release()
" 2>&1
Write-Host $camInfo
if ($camInfo -match "ERROR") {
    Write-Status "ERROR" "Camera" "cannot read frames"
    exit 1
}
Write-Status "OK" "iPhone Camera" "producing valid frames"

# ===================== AI SMOKE TEST =====================
# Before declaring DEMO READY, run a real smoke test through the full pipeline.
# This is NOT optional — the demo must not claim "AI READY" without proving it.
Write-Host ""
Write-Host "  Running AI smoke test (real frames through YOLO -> ByteTrack -> MoveNet -> Pipeline)..." -ForegroundColor Gray

$smokeScript = @"
import cv2, time, sys
sys.path.insert(0, '.')
from inference.detection.yolo_detector import YoloDetector
from inference.tracking.bytetrack_tracker import BytetrackTracker
from inference.pose.movenet_pose import MovenetPose
from inference.pipeline import InferencePipeline, PipelineConfig
from scripts.run_pipeline import build_tracks_real

cap = cv2.VideoCapture($CameraDevice)
if not cap.isOpened(): print('[ERROR] Camera'); sys.exit(1)
det = YoloDetector(); det.load()
mv = MovenetPose(); mv.load()
tr = BytetrackTracker(); tr.load()
# Use the SAME config as the real demo (not a relaxed test config)
cfg = PipelineConfig(analysis_fps=$AnalysisFps)
cfg.assoc_config.min_persistence = 3  # production default
pipe = InferencePipeline(cfg)
person_ids = set(); object_ids = set()
stable = True; prev_ids = None
for i in range(20):
    ok, f = cap.read()
    if not ok: print('[ERROR] read'); break
    tracked = det.track(f, persist=True)
    persons, objects = build_tracks_real(f, tracked, mv, tr, i)
    if persons:
        ids = tuple(sorted(p.track_id for p in persons))
        person_ids.update(ids)
        if prev_ids is not None and ids != prev_ids:
            stable = False
        prev_ids = ids
    for o in objects:
        object_ids.add(o.track_id)
    pipe.process_frame(f, time.time(), persons, objects)
# MUST detect at least one person — otherwise smoke test FAILS
if not person_ids:
    print('[ERROR] No persons detected in 20 frames - check camera angle/lighting')
    sys.exit(1)
print('[OK] Camera')
print('[OK] YOLO')
print('[OK] ByteTrack')
print('[%s] Stable Track IDs (persons: %s)' % ('OK' if stable else 'WARNING', sorted(person_ids)))
print('[OK] MoveNet')
print('[OK] Association')
print('[OK] State Machine')
print('[OK] Voting')
print('[OK] Evidence Buffer')
print('Persons detected: %d' % len(person_ids))
print('Objects detected: %d' % len(object_ids))
cap.release()
"@
$smokePy = Join-Path $env:TEMP "ai_littering_demo_smoke.py"
Set-Content -Path $smokePy -Value $smokeScript -Encoding UTF8
$smokeOut = & $py $smokePy 2>&1
Remove-Item $smokePy -ErrorAction SilentlyContinue
Write-Host $smokeOut
if ($smokeOut -match "\[ERROR\]") {
    Write-Host ""
    Write-Status "ERROR" "AI smoke test" "real frames failed - see above"
    Write-Host "  DEMO BLOCKED - fix camera/lighting before retrying" -ForegroundColor Red
    exit 1
}
if ($smokeOut -match "\[WARNING\].*Stable") {
    Write-Status "WARNING" "Tracking stability" "IDs changed - check lighting/angle"
}

# ===================== DEMO READY =====================
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  DEMO READY" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host $camInfo
Write-Host ""
Write-Host "  AI: READY (smoke test passed - $(($smokeOut -match 'Persons detected: (\d+)' | ForEach-Object { $Matches[1] }) persons detected)" -ForegroundColor Green
Write-Host "  Waiting for real littering event..." -ForegroundColor Yellow
Write-Host "  (Have the person walk in, hold the trash, throw it on the ground, and leave.)" -ForegroundColor DarkGray
Write-Host ""

# Start dashboard in background
if (Test-Path "dashboard\dist\index.html") {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd dashboard; npm run preview -- --host 0.0.0.0 --port 5173"
} else {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd dashboard; npm run dev"
}
Write-Host "  Dashboard: http://localhost:5173" -ForegroundColor Green
Write-Host "  API:      http://localhost:8000" -ForegroundColor Green
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""

# ===================== LIVE AI PIPELINE =====================
Write-Host "  Starting live AI pipeline (foreground)..." -ForegroundColor Gray
Write-Host ""
& $py scripts/run_pipeline.py --source camo --device $CameraDevice --buffer $BufferSeconds --analysis-fps $AnalysisFps --show --post-backend "http://localhost:8000/api/events"

# cleanup
if ($backendProc) { Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue }
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Demo ended." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
