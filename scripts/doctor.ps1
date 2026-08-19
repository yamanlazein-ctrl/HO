<#
.SYNOPSIS
    AI Littering Detection — Deep diagnostic (doctor) script.
.DESCRIPTION
    Reports system health in four sections:
      • System          — OS, Python, Node, npm, git, venv
      • AI              — ultralytics, cv2, tensorflow, model weights, pipeline
      • Infrastructure  — Docker, PostgreSQL, Backend, Dashboard
      • Camera          — real cv2.VideoCapture probing via camera_discovery.py

    Camera rules:
      - NEVER reports [OK] unless cv2.VideoCapture actually opens and reads
        a frame from a real device.
      - Uses camera_discovery.py for genuine multi-device probing.
      - If no camera is found, prints [WAITING FOR CAMERA] with instructions
        — this is expected on machines without a connected webcam or iPhone
        virtual camera (Camo/Iriun).

    Exit codes:
      0  — no critical ERRORs (warnings/waiting items are non-blocking)
      1  — one or more critical ERRORs (e.g. best.pt missing, Python missing)

.NOTES
    File     : scripts/doctor.ps1
    Platform : Windows PowerShell 5.1+ / PowerShell 7+
    Verified : UNVERIFIED — PowerShell script, NOT executed in sandbox
               (no PowerShell runtime). Written carefully per spec.
#>

#Requires -Version 5.1
[CmdletBinding()]
param()

$RepoRoot   = Split-Path $PSScriptRoot -Parent
$venvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) { $venvPython = 'python' }  # fallback

$script:Errors   = 0
$script:Warnings = 0
$script:OKs      = 0
$script:Waiting  = 0
$script:Blocking = [System.Collections.Generic.List[string]]::new()

function Write-Diag {
    param([string]$Tag, [string]$Name, [string]$Detail = "")
    $colour = switch ($Tag) {
        'OK'      { 'Green'  }
        'WARNING' { 'Yellow' }
        'ERROR'   { 'Red'    }
        'WAITING' { 'Cyan'   }
        default   { 'White'  }
    }
    $line = "  [$Tag] $Name"
    if ($Detail) { $line += " — $Detail" }
    Write-Host $line -ForegroundColor $colour
    switch ($Tag) {
        'OK'      { $script:OKs++ }
        'WARNING' { $script:Warnings++ }
        'ERROR'   { $script:Errors++ }
        'WAITING' { $script:Waiting++ }
    }
}

function Diag-Error {
    param([string]$Name, [string]$Detail = "", [switch]$Blocking)
    Write-Diag 'ERROR' $Name $Detail
    if ($Blocking) { $script:Blocking.Add("[$Name] $Detail") }
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# Load .env
$envFile = Join-Path $RepoRoot '.env'
$envData = @{}
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^(\w+)\s*=\s*(.+)$') { $envData[$Matches[1]] = $Matches[2].Trim() }
    }
}
$dbUrl = if ($envData['DATABASE_URL']) { $envData['DATABASE_URL'] } else { 'postgresql://litter:litter@localhost:5432/littering' }
$backendPort = if ($envData['BACKEND_PORT']) { [int]$envData['BACKEND_PORT'] } else { 8000 }
$dashPort = if ($envData['DASHBOARD_PORT']) { [int]$envData['DASHBOARD_PORT'] } else { 5173 }

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  AI Littering Detection — Doctor" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# ══════════════════════════════════════════════════════════════════════════
# Section 1: SYSTEM
# ══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "── System ──" -ForegroundColor White

# OS
$os = if ($PSVersionTable.Platform -and $PSVersionTable.Platform -ne 'Win32NT') {
    $PSVersionTable.Platform
} else { "Windows" }
if ($os -eq "Windows") {
    Write-Diag 'OK' "OS" $os
} else {
    Write-Diag 'WARNING' "OS" "$os — this system targets Windows; running on non-Windows"
}

# Python
$pyOk = $false
if (Test-Command 'python') {
    $pyOut = & python --version 2>&1
    if ($pyOut -match 'Python\s+(\d+)\.(\d+)') {
        $maj, $min = [int]$Matches[1], [int]$Matches[2]
        if ($maj -ge 3 -and $min -ge 9) {
            Write-Diag 'OK' "Python" "$($Matches[1]).$($Matches[2])"
            $pyOk = $true
        } else {
            Diag-Error "Python" "version $($Matches[1]).$($Matches[2]) — need >= 3.9" -Blocking
        }
    } else {
        Diag-Error "Python" "cannot parse version" -Blocking
    }
} else {
    Diag-Error "Python" "not on PATH" -Blocking
}

# venv
if (Test-Path (Join-Path $RepoRoot '.venv\Scripts\python.exe')) {
    Write-Diag 'OK' "Python venv" ".venv exists"
} else {
    Write-Diag 'WARNING' "Python venv" ".venv not found — run .\install.ps1"
}

# Node
if (Test-Command 'node') {
    $nv = (& node --version 2>&1).ToString().Trim()
    if ($nv -match 'v(\d+)') {
        if ([int]$Matches[1] -ge 18) {
            Write-Diag 'OK' "Node" $nv
        } else {
            Write-Diag 'WARNING' "Node" "$nv — need >= 18"
        }
    } else { Write-Diag 'WARNING' "Node" "version parse failed" }
} else { Write-Diag 'WARNING' "Node" "not on PATH" }

# npm
if (Test-Command 'npm') {
    Write-Diag 'OK' "npm" (& npm --version 2>&1 | Select-Object -First 1)
} else { Write-Diag 'WARNING' "npm" "not on PATH" }

# git
if (Test-Command 'git') {
    Write-Diag 'OK' "git" (& git --version 2>&1 | Select-Object -First 1)
} else { Write-Diag 'WARNING' "git" "not on PATH" }


# ══════════════════════════════════════════════════════════════════════════
# Section 2: AI
# ══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "── AI ──" -ForegroundColor White

# AI imports + pipeline smoke test in one subprocess
$aiScript = @"
import sys
sys.path.insert(0, r'$RepoRoot')
results = []
# ultralytics
try:
    import ultralytics
    results.append(('ultralytics', 'OK', ultralytics.__version__))
except Exception as e:
    results.append(('ultralytics', 'FAIL', str(e)))
# cv2
try:
    import cv2
    results.append(('cv2', 'OK', cv2.__version__))
except Exception as e:
    results.append(('cv2', 'FAIL', str(e)))
# tensorflow
try:
    import tensorflow as tf
    results.append(('tensorflow', 'OK', tf.__version__))
except Exception as e:
    results.append(('tensorflow', 'FAIL', str(e)))
# pipeline
try:
    from inference.pipeline import InferencePipeline, PipelineConfig
    InferencePipeline(PipelineConfig())
    results.append(('pipeline', 'OK', 'instantiated'))
except Exception as e:
    results.append(('pipeline', 'FAIL', str(e)))
for name, status, info in results:
    print(f'{name}|{status}|{info}')
"@

$env:PYTHONPATH = $RepoRoot
$aiResult = & $venvPython -c $aiScript 2>&1
foreach ($line in ($aiResult -split "`n")) {
    $line = $line.Trim()
    if ($line -match '^(\w+)\|(\w+)\|(.+)$') {
        $name, $status, $info = $Matches[1], $Matches[2], $Matches[3]
        if ($status -eq 'OK') {
            Write-Diag 'OK' "AI: $name" $info
        } else {
            Write-Diag 'ERROR' "AI: $name" $info
        }
    }
}

# Model weights
$bestPt = Join-Path $RepoRoot 'inference\detection\weights\best.pt'
if (Test-Path $bestPt) {
    $sz = [math]::Round((Get-Item $bestPt).Length / 1MB, 1)
    Write-Diag 'OK' "best.pt (litter model)" "$sz MB"

    # Validate model classes against AssociationConfig
    $classScript = @"
import sys
sys.path.insert(0, r'$RepoRoot')
try:
    from ultralytics import YOLO
    model = YOLO(r'$bestPt')
    model_classes = sorted([str(v).lower() for v in model.names.values()])
    print('CLASSES:' + ','.join(model_classes))
except Exception as e:
    print(f'LOAD_ERROR:{type(e).__name__}:{e}')
"@
    $classResult = & $venvPython -c $classScript 2>&1
    $classStr = ($classResult | Where-Object { $_ -match '^(CLASSES:|LOAD_ERROR:)' } | Select-Object -First 1)
    if ($classStr) { $classStr = $classStr.ToString().Trim() }

    if ($classStr -match '^CLASSES:(.+)$') {
        $modelClassList = $Matches[1] -split ',' | ForEach-Object { $_.Trim() }
        Write-Diag 'OK' "best.pt classes" ($modelClassList -join ', ')

        # Compare against AssociationConfig
        $assocScript = @"
import sys
sys.path.insert(0, r'$RepoRoot')
try:
    from inference.association.person_object_assoc import AssociationConfig
    classes = list(AssociationConfig().litter_candidate_classes)
    print('ASSOC:' + ','.join(classes))
except Exception as e:
    print(f'ASSOC_ERROR:{type(e).__name__}:{e}')
"@
        $assocResult = & $venvPython -c $assocScript 2>&1
        $assocStr = ($assocResult | Where-Object { $_ -match '^(ASSOC:|ASSOC_ERROR:)' } | Select-Object -First 1)
        if ($assocStr) { $assocStr = $assocStr.ToString().Trim() }

        if ($assocStr -match '^ASSOC:(.+)$') {
            $assocClasses = $Matches[1] -split ',' | ForEach-Object { $_.Trim().ToLower() }
            $missingInAssoc = @(); $missingInModel = @()
            foreach ($mc in $modelClassList) { if ($mc -notin $assocClasses) { $missingInAssoc += $mc } }
            foreach ($ac in $assocClasses)   { if ($ac -notin $modelClassList) { $missingInModel += $ac } }
            if ($missingInAssoc.Count -eq 0 -and $missingInModel.Count -eq 0) {
                Write-Diag 'OK' "Class validation" "model classes match AssociationConfig"
            } else {
                $d = ""
                if ($missingInAssoc.Count -gt 0) { $d += "In model, NOT in config: $($missingInAssoc -join ', '). " }
                if ($missingInModel.Count -gt 0) { $d += "In config, NOT in model: $($missingInModel -join ', ')." }
                Write-Diag 'WARNING' "Class validation" $d.Trim()
            }
        } else {
            Write-Diag 'WARNING' "Class validation" "could not load AssociationConfig"
        }
    } elseif ($classStr -match '^LOAD_ERROR:(.+)$') {
        Write-Diag 'ERROR' "best.pt load" "Failed to load: $($Matches[1])"
    } else {
        Write-Diag 'WARNING' "best.pt classes" "could not parse model classes"
    }
} else {
    Diag-Error "best.pt (litter model)" @"
MISSING — download from reference repo (Anti-Littering-System-Computer-Vision, MIT)
  and place at inference/detection/weights/best.pt
"@ -Blocking
}

$yolov8n = Join-Path $RepoRoot 'yolov8n.pt'
if (Test-Path $yolov8n) {
    $sz = [math]::Round((Get-Item $yolov8n).Length / 1MB, 1)
    Write-Diag 'OK' "yolov8n.pt (person model)" "$sz MB"
} else {
    Write-Diag 'WARNING' "yolov8n.pt (person model)" "not in repo — ultralytics auto-downloads on first use"
}


# ══════════════════════════════════════════════════════════════════════════
# Section 3: INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "── Infrastructure ──" -ForegroundColor White

# Docker
if (Test-Command 'docker') {
    $null = & docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Diag 'OK' "Docker" "daemon running"
    } else {
        Write-Diag 'WARNING' "Docker" "CLI present but daemon not running"
    }
} else {
    Write-Diag 'WARNING' "Docker" "not installed — Postgres needs Docker or external DB"
}

# Database
$dbScript = @"
import os, sys
sys.path.insert(0, r'$RepoRoot')
url = os.environ.get('DATABASE_URL', r'$dbUrl')
try:
    from sqlalchemy import create_engine, text
    eng = create_engine(url, connect_args={'connect_timeout': 5})
    with eng.connect() as c:
        c.execute(text('SELECT 1'))
    print('DB_OK')
except Exception as e:
    print(f'DB_FAIL:{type(e).__name__}:{e}')
"@

$env:PYTHONPATH = $RepoRoot
$env:DATABASE_URL = $dbUrl
$dbResult = & $venvPython -c $dbScript 2>&1
$dbStr = $dbResult.ToString().Trim()
if ($dbStr -match 'DB_OK') {
    Write-Diag 'OK' "Database" "connected"
} elseif ($dbStr -match 'DB_FAIL:(.+)') {
    Write-Diag 'WARNING' "Database" "cannot connect: $($Matches[1]) — start Docker Postgres or set DATABASE_URL"
} else {
    Write-Diag 'WARNING' "Database" "could not determine status"
}

# Backend
$backendOk = $false
try {
    $resp = Invoke-RestMethod -Uri "http://localhost:$backendPort/health" -Method GET -TimeoutSec 5 -ErrorAction Stop
    if ($resp.status -eq 'ok') {
        Write-Diag 'OK' "Backend /health" "responding on port $backendPort"
        $backendOk = $true
    } else {
        Write-Diag 'WARNING' "Backend /health" "unexpected response: $resp"
    }
} catch {
    Write-Diag 'WAITING' "Backend /health" "not running on port $backendPort — start: uvicorn backend.main:app --port $backendPort"
}

if ($backendOk) {
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:$backendPort/api/status" -Method GET -TimeoutSec 5 -ErrorAction Stop
        $aiStatus = $resp.ai_engine.status
        $camStat = $resp.camera.status
        Write-Diag 'OK' "Backend /api/status" "ai_engine=$aiStatus, camera=$camStat"
    } catch {
        Write-Diag 'WARNING' "Backend /api/status" "endpoint responded with error"
    }
}

# Dashboard
$distPath = Join-Path $RepoRoot 'dashboard\dist'
$nodeMods = Join-Path $RepoRoot 'dashboard\node_modules'
if (Test-Path $nodeMods) {
    Write-Diag 'OK' "Dashboard node_modules" "installed"
} else {
    Write-Diag 'WARNING' "Dashboard node_modules" "not installed — run: cd dashboard && npm install"
}
if (Test-Path $distPath) {
    $fc = (Get-ChildItem $distPath -Recurse -File).Count
    Write-Diag 'OK' "Dashboard dist/ (build)" "$fc files"
} else {
    Write-Diag 'WARNING' "Dashboard dist/ (build)" "not built — run: cd dashboard && npm run build"
}
# Dev server
try {
    $r = Invoke-WebRequest -Uri "http://localhost:$dashPort" -Method GET -TimeoutSec 5 -ErrorAction Stop -UseBasicParsing
    if ($r.StatusCode -eq 200) {
        Write-Diag 'OK' "Dashboard dev server" "responding on port $dashPort"
    }
} catch {
    Write-Diag 'WAITING' "Dashboard dev server" "not running on port $dashPort — start: cd dashboard && npm run dev"
}


# ══════════════════════════════════════════════════════════════════════════
# Section 4: CAMERA
# ══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "── Camera ──" -ForegroundColor White

# Use camera_discovery.py for REAL multi-device probing.
# NEVER report [OK] unless a real device opens and reads a frame.
$camScript = @"
import sys, os
sys.path.insert(0, r'$RepoRoot')
try:
    from scripts.camera_discovery import discover_cameras
    cams = discover_cameras(max_idx=5, probe_frames=8)
    live = [c for c in cams if c.status.value == 'LIVE']
    found = [c for c in cams if c.status.value != 'NOT_FOUND']
    if live:
        c = live[0]
        print(f'LIVE:{c.index}:{c.resolution_str()}:{c.fps}')
    elif found:
        statuses = ','.join(f'{c.index}:{c.status.value}' for c in found)
        print(f'FOUND_BUT_BAD:{statuses}')
    else:
        print('NO_CAMERA')
except ImportError as e:
    print(f'CV2_MISSING:{e}')
except Exception as e:
    print(f'CAM_ERROR:{type(e).__name__}:{e}')
"@

$env:PYTHONPATH = $RepoRoot
$camResult = & $venvPython -c $camScript 2>&1
# Filter out OpenCV stderr warnings, keep only our structured output
$camStr = ""
foreach ($line in ($camResult -split "`n")) {
    $line = $line.Trim()
    if ($line -match '^(LIVE:|FOUND_BUT_BAD:|NO_CAMERA|CV2_MISSING:|CAM_ERROR:)') {
        $camStr = $line
        break
    }
}

if ($camStr -match '^LIVE:(\d+):(\d+x\d+):([\d.]+)') {
    Write-Diag 'OK' "Camera (device $($Matches[1]))" "LIVE, resolution $($Matches[2]), fps $($Matches[3])"
} elseif ($camStr -match '^FOUND_BUT_BAD:(.+)') {
    Write-Diag 'WARNING' "Camera" "device(s) found but not LIVE: $($Matches[1])"
} elseif ($camStr -match '^NO_CAMERA') {
    Write-Diag 'WAITING' "Camera" @"
WAITING FOR CAMERA — no device opened.
  • Connect a USB webcam or iPhone (via Camo/Iriun virtual camera).
  • On Windows, check Privacy settings → Allow apps to access your camera.
  • Try a different device index: set CAMERA_DEVICE_INDEX=1 in .env
  • Re-run: python scripts/camera_discovery.py
"@
} elseif ($camStr -match '^CV2_MISSING:(.+)') {
    Write-Diag 'WARNING' "Camera" "OpenCV not installed: $($Matches[1])"
} elseif ($camStr -match '^CAM_ERROR:(.+)') {
    Write-Diag 'WAITING' "Camera" "error probing devices: $($Matches[1])"
} else {
    Write-Diag 'WAITING' "Camera" "could not determine camera status — run: python scripts/camera_discovery.py"
}


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  DOCTOR SUMMARY" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  OK:       $script:OKs" -ForegroundColor Green
Write-Host "  WARNING:  $script:Warnings" -ForegroundColor Yellow
Write-Host "  WAITING:  $script:Waiting (informational)" -ForegroundColor Cyan
Write-Host "  ERROR:    $script:Errors" -ForegroundColor Red
Write-Host "==========================================" -ForegroundColor Cyan

if ($script:Errors -gt 0) {
    Write-Host ""
    if ($script:Blocking.Count -gt 0) {
        Write-Host "  BLOCKING ERRORS (must fix):" -ForegroundColor Red
        $script:Blocking | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    }
    Write-Host ""
    Write-Host "  ❌ $script:Errors error(s) found — fix them before running the system." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  ✅ No errors. Warnings/waiting items are non-blocking." -ForegroundColor Green
exit 0
