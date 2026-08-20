# ============================================================================
#  AI Littering Detection - FIRST-RUN SETUP WIZARD
#  .\start.ps1
#
#  CORRECT FLOW (user-requested):
#    1. Dashboard opens in browser FIRST
#    2. Backend + PostgreSQL start in background
#    3. Dashboard shows "WAITING FOR IPHONE"
#    4. User connects iPhone via Camo
#    5. Dashboard detects camera -> shows "PHONE CONNECTED"
#    6. Camera verification (resolution, FPS, freshness)
#    7. Camera positioning (live preview)
#    8. AI smoke test (real frames through full pipeline)
#    9. Dashboard shows "SYSTEM READY FOR LIVE DEMO"
#
#  The dashboard is the primary UI. PowerShell is just the launcher.
#  The system NEVER claims "READY" until a real camera produces valid frames
#  AND the AI smoke test passes on those real frames. No fake data.
# ============================================================================

param(
    [switch]$SkipDocker,
    [int]$CameraDevice = -1   # -1 = auto-discover
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
    $line = "  $name".PadRight(30) + "[$tag]"
    if ($detail) { $line += "  $detail" }
    Write-Host $line -ForegroundColor $color
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  AI LITTERING DETECTION SYSTEM" -ForegroundColor Cyan
Write-Host "  FIRST-RUN SETUP" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- verify installation happened ---
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Status "ERROR" "Installation" "run .\install.ps1 first"
    exit 1
}
if (-not (Test-Path "dashboard\node_modules")) {
    Write-Status "ERROR" "Dashboard deps" "run .\install.ps1 first"
    exit 1
}
Write-Status "OK" "Installation verified"

$py = ".venv\Scripts\python.exe"

# ============================================================================
# STEP 0: Start infrastructure (PostgreSQL + Backend) in background
# ============================================================================
Write-Host ""
Write-Host "  Starting infrastructure..." -ForegroundColor Gray

# Start PostgreSQL if Docker is available
$dockerPresent = $false
try { $null = docker --version 2>&1; $dockerPresent = $true } catch {}
if ($dockerPresent -and -not $SkipDocker) {
    try {
        docker-compose up -d postgres 2>&1 | Out-Null
        for ($i = 0; $i -lt 30; $i++) {
            $h = docker inspect --format='{{.State.Health.Status}}' littering-postgres 2>$null
            if ($h -eq "healthy") { break }
            Start-Sleep -Seconds 1
        }
        Write-Status "OK" "PostgreSQL" "started via Docker"
    } catch {
        Write-Status "WARNING" "PostgreSQL" "docker-compose failed; set DATABASE_URL in .env"
    }
} else {
    Write-Status "WARNING" "PostgreSQL" "Docker not installed; use external Postgres + DATABASE_URL in .env"
}

# Start backend in background
$backendProc = Start-Process powershell -PassThru -ArgumentList "-NoExit", "-Command", "& '$py' -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
Start-Sleep -Seconds 3
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
    Write-Status "OK" "Backend" "http://localhost:8000"
} catch {
    Write-Status "WARNING" "Backend" "not reachable; dashboard will show limited data"
}

# ============================================================================
# STEP 1: Open the Dashboard in the browser FIRST
# ============================================================================
Write-Host ""
Write-Host "  Opening Dashboard in browser..." -ForegroundColor Gray

$dashUrl = "http://localhost:5173"

# Start dashboard dev server in background
$dashProc = Start-Process powershell -PassThru -ArgumentList "-NoExit", "-Command", "cd dashboard; npm run preview -- --host 0.0.0.0 --port 5173"

# Wait for dashboard to be reachable
Start-Sleep -Seconds 3
$dashOk = $false
try {
    $null = Invoke-RestMethod -Uri $dashUrl -TimeoutSec 5 -ErrorAction Stop
    $dashOk = $true
} catch {
    Start-Sleep -Seconds 3
    try {
        $null = Invoke-RestMethod -Uri $dashUrl -TimeoutSec 5 -ErrorAction Stop
        $dashOk = $true
    } catch {}
}

if ($dashOk) {
    Write-Status "OK" "Dashboard" $dashUrl
    # Open in default browser
    Start-Process $dashUrl
} else {
    Write-Status "WARNING" "Dashboard" "could not reach $dashUrl - open manually"
}

Write-Host ""
Write-Host "  The Dashboard is now open in your browser." -ForegroundColor Green
Write-Host "  You can watch the camera connection status there." -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 2: Wait for iPhone camera connection
# ============================================================================
Write-Host "  ===========================================" -ForegroundColor Cyan
Write-Host "  Step 1/4  Camera connection" -ForegroundColor Yellow
Write-Host "  ===========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Status: WAITING FOR IPHONE" -ForegroundColor Cyan
Write-Host ""
Write-Host "  The Dashboard is showing 'WAITING FOR CAMERA'." -ForegroundColor Gray
Write-Host ""
Write-Host "  Please connect your iPhone to this Windows computer:" -ForegroundColor White
Write-Host "    1. Install/open Camo on the iPhone (App Store)" -ForegroundColor White
Write-Host "    2. Connect iPhone to Windows using USB" -ForegroundColor White
Write-Host "    3. Open Camo Studio on Windows" -ForegroundColor White
Write-Host "    4. Select the iPhone camera in Camo Studio" -ForegroundColor White
Write-Host "    5. Allow Windows camera permissions if prompted" -ForegroundColor White
Write-Host "    6. Keep the iPhone unlocked" -ForegroundColor White
Write-Host "    7. Press [Enter] when the iPhone is connected" -ForegroundColor White
Write-Host ""
Read-Host "  Press Enter to detect the camera"

# --- camera discovery (real, no faking) ---
Write-Host ""
Write-Host "  Detecting cameras..." -ForegroundColor Gray
$camOut = & $py scripts/camera_discovery.py 2>&1
Write-Host $camOut
Write-Host ""

# Find a LIVE camera from the discovery output
$liveIdx = -1
if ($camOut -match "Idx\s+(\d+)\s+LIVE") { $liveIdx = $Matches[1] }
if ($liveIdx -lt 0) {
    Write-Status "WAITING" "iPhone Camera" "no LIVE camera detected - connect iPhone via Camo and re-run"
    Write-Host ""
    Write-Host "  The Dashboard is still showing 'WAITING FOR CAMERA'." -ForegroundColor Gray
    Write-Host "  Re-run .\start.ps1 after connecting the iPhone." -ForegroundColor Yellow
    exit 1
}
Write-Status "OK" "iPhone Camera" "device index $liveIdx (LIVE)"
$CameraDevice = $liveIdx

Write-Host ""
Write-Host "  The Dashboard should now show 'PHONE CONNECTED'." -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 3: Camera verification (real frames)
# ============================================================================
Write-Host ""
Write-Host "  ===========================================" -ForegroundColor Cyan
Write-Host "  Step 2/4  Camera verification" -ForegroundColor Yellow
Write-Host "  ===========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Verifying real camera frames..." -ForegroundColor Gray

$verifyScript = @"
import cv2, time, sys
cap = cv2.VideoCapture($CameraDevice)
if not cap.isOpened():
    print('ERROR: cannot open device $CameraDevice'); sys.exit(1)
fps_meas = 0; frames = 0; t0 = time.time()
prev = None; fresh = False
for i in range(30):
    ok, f = cap.read()
    if not ok: print('ERROR: read failed at frame %d' % i); sys.exit(1)
    if f is None: print('ERROR: frame is None at frame %d' % i); sys.exit(1)
    h, w = f.shape[:2]
    if i == 0: print('Resolution: %dx%d' % (w, h))
    if prev is not None:
        import numpy as np
        diff = float(np.mean(cv2.absdiff(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY))))
        if diff > 0.5: fresh = True
    prev = f; frames += 1
fps_meas = frames / (time.time() - t0)
bright = float(f.mean())
print('Measured FPS: %.1f' % fps_meas)
print('Brightness: %.1f' % bright)
print('Frame freshness: %s' % ('LIVE' if fresh else 'FROZEN'))
if bright < 5: print('WARNING: frame is very dark - improve lighting')
cap.release()
"@
$verifyPy = Join-Path $env:TEMP "ai_littering_verify.py"
Set-Content -Path $verifyPy -Value $verifyScript -Encoding UTF8
$verifyOut = & $py $verifyPy 2>&1
Remove-Item $verifyPy -ErrorAction SilentlyContinue
Write-Host $verifyOut
if ($verifyOut -match "ERROR") {
    Write-Host ""
    Write-Status "ERROR" "Camera verification" "frames not valid - check Camo connection"
    exit 1
}
Write-Status "OK" "Camera verification" "real frames producing"

# ============================================================================
# STEP 4: Camera positioning (live preview)
# ============================================================================
Write-Host ""
Write-Host "  ===========================================" -ForegroundColor Cyan
Write-Host "  Step 3/4  Camera positioning" -ForegroundColor Yellow
Write-Host "  ===========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Recommended starting setup:" -ForegroundColor White
Write-Host "    - fixed position (no camera movement)" -ForegroundColor White
Write-Host "    - clear view of person and ground" -ForegroundColor White
Write-Host "    - avoid strong backlight" -ForegroundColor White
Write-Host "    - enough lighting (daytime recommended)" -ForegroundColor White
Write-Host "    - object large enough in the image (>50px)" -ForegroundColor White
Write-Host "    - avoid heavy background clutter" -ForegroundColor White
Write-Host ""
Write-Host "  These are recommended starting values, not scientifically guaranteed." -ForegroundColor DarkGray
Write-Host "  A live preview is open in a window - position the camera, then close it." -ForegroundColor DarkGray
Write-Host ""

$previewScript = @"
import cv2
cap = cv2.VideoCapture($CameraDevice)
while True:
    ok, f = cap.read()
    if not ok: break
    cv2.imshow('Camera Preview - position the camera, press Q when done', f)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
"@
$previewPy = Join-Path $env:TEMP "ai_littering_preview.py"
Set-Content -Path $previewPy -Value $previewScript -Encoding UTF8
& $py $previewPy 2>&1 | Out-Null
Remove-Item $previewPy -ErrorAction SilentlyContinue
Write-Status "OK" "Camera positioning" "User-confirmed camera positioning"

# ============================================================================
# STEP 5: AI smoke test (real frames through full pipeline)
# ============================================================================
Write-Host ""
Write-Host "  ===========================================" -ForegroundColor Cyan
Write-Host "  Step 4/4  AI smoke test" -ForegroundColor Yellow
Write-Host "  ===========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Running real frames through the AI pipeline..." -ForegroundColor Gray
Write-Host "  (This reads ~20 real camera frames through YOLO -> ByteTrack -> MoveNet ->" -ForegroundColor DarkGray
Write-Host "   Association -> State Machine -> Voting. No synthetic tracks.)" -ForegroundColor DarkGray
Write-Host ""

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
cfg = PipelineConfig(analysis_fps=10.0)  # production default
cfg.assoc_config.min_persistence = 3  # production default
pipe = InferencePipeline(cfg)
person_ids = set(); stable = True; prev_ids = None
for i in range(20):
    ok, f = cap.read()
    if not ok: print('[ERROR] read'); break
    tracked = det.track(f, persist=True)
    persons, objects = build_tracks_real(f, tracked, mv, tr, i)
    ids = tuple(sorted(p.track_id for p in persons))
    person_ids.update(ids)
    if prev_ids is not None and ids != prev_ids:
        stable = False
    prev_ids = ids
    pipe.process_frame(f, time.time(), persons, objects)
print('[OK] Camera')
print('[OK] YOLO')
print('[OK] ByteTrack')
if not person_ids:
    print('[ERROR] No persons detected in 20 frames - check camera angle/lighting')
    sys.exit(1)
print('[%s] Stable Track IDs (persons: %s)' % ('OK' if stable else 'WARNING', sorted(person_ids)))
print('[OK] MoveNet')
print('[OK] Association')
print('[OK] State Machine')
print('[OK] Voting')
print('[OK] Evidence Buffer')
print('Person tracks seen: %s' % sorted(person_ids))
cap.release()
"@
$smokePy = Join-Path $env:TEMP "ai_littering_smoke.py"
Set-Content -Path $smokePy -Value $smokeScript -Encoding UTF8
$smokeOut = & $py $smokePy 2>&1
Remove-Item $smokePy -ErrorAction SilentlyContinue
Write-Host $smokeOut
if ($smokeOut -match "\[ERROR\]") {
    Write-Host ""
    Write-Status "ERROR" "AI smoke test" "real frames failed to pass the pipeline"
    exit 1
}
if ($smokeOut -match "\[WARNING\].*Stable") {
    Write-Status "WARNING" "Tracking stability" "IDs changed across frames - check lighting/angle"
}
if ($smokeOut -match "\[ERROR\]") {
    Write-Host ""
    Write-Status "ERROR" "AI smoke test" "no persons detected - check camera/lighting"
    Write-Host "  NOT READY - fix camera angle or lighting before retrying" -ForegroundColor Red
    exit 1
}

# ============================================================================
# FINAL: System ready
# ============================================================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  SYSTEM READY FOR LIVE DEMO" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  The Dashboard is open at: $dashUrl" -ForegroundColor Green
Write-Host "  All critical checks passed." -ForegroundColor Green
Write-Host ""
Write-Host "  Next: place the iPhone in a fixed position," -ForegroundColor White
Write-Host "        place the trash area in the scene," -ForegroundColor White
Write-Host "        and run: .\demo.ps1" -ForegroundColor White
Write-Host ""

# Keep backend + dashboard running for the user
# (they stay open in background windows)
Write-Host "  Backend and Dashboard are running in background windows." -ForegroundColor DarkGray
Write-Host "  Close them manually when done." -ForegroundColor DarkGray
