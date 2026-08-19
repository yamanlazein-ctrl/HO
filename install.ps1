<#
.SYNOPSIS
    AI Littering Detection — Canonical Windows installer.

.DESCRIPTION
    The ONE canonical Windows install command:

        .\install.ps1

    This script verifies prerequisites, creates the Python venv, installs
    all deps, checks key imports, verifies model weights, starts PostgreSQL
    (if Docker is available), initialises the DB schema, runs a backend
    health check, builds the frontend, runs the test suite, and performs
    a minimal inference smoke test — then validates model classes and
    scans the repo for fake/mock/placeholder code.

    Every step prints [OK] / [WARNING] / [ERROR] with the step name.
    It NEVER fakes success, NEVER silently continues after an ERROR on a
    critical prerequisite, and does NOT start the AI pipeline or live demo.

    Exit codes:
        0  — zero ERRORs (PASS)
        1  — one or more critical ERRORs (FAIL)

.NOTES
    File     : install.ps1
    Location : repo root (canonical installer)
    Platform : Windows PowerShell 5.1+ / PowerShell 7+
    Verified : UNVERIFIED — this is a PowerShell script and was NOT
               executed in the build sandbox (no PowerShell runtime).
               Written carefully per spec.
#>

#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$SkipDocker   # skip Docker/PostgreSQL steps entirely
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

$script:OK_COUNT       = 0
$script:WARN_COUNT     = 0
$script:ERROR_COUNT    = 0
$script:ERROR_MESSAGES = [System.Collections.Generic.List[string]]::new()
$script:BLOCKING       = [System.Collections.Generic.List[string]]::new()

function Write-Step {
    param([string]$Tag, [string]$Name, [string]$Detail = "")
    $colour = switch ($Tag) {
        'OK'      { 'Green'  }
        'WARNING' { 'Yellow' }
        'ERROR'   { 'Red'    }
        default   { 'White'  }
    }
    $line = "[$Tag] $Name"
    if ($Detail) { $line += " - $Detail" }
    Write-Host $line -ForegroundColor $colour
}

function Step-OK {
    param([string]$Name, [string]$Detail = "")
    $script:OK_COUNT++
    Write-Step 'OK' $Name $Detail
}

function Step-Warn {
    param([string]$Name, [string]$Detail = "")
    $script:WARN_COUNT++
    Write-Step 'WARNING' $Name $Detail
}

function Step-Error {
    param([string]$Name, [string]$Detail = "", [switch]$Blocking)
    $script:ERROR_COUNT++
    $msg = "[$Name] $Detail"
    $script:ERROR_MESSAGES.Add($msg)
    if ($Blocking) { $script:BLOCKING.Add($msg) }
    Write-Step 'ERROR' $Name $Detail
}

# Stop immediately on blocking error.
function Stop-OnBlocking {
    if ($script:BLOCKING.Count -gt 0) {
        Write-Host ""
        Write-Host "FATAL: Blocking error(s) detected — cannot continue." -ForegroundColor Red
        $script:BLOCKING | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        exit 1
    }
}

# Returns $true if a command exists on PATH.
function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# Returns the major version number of a tool that supports --version.
function Get-VersionMajor {
    param([string]$Cmd)
    try {
        $out = & $Cmd --version 2>&1 | Select-Object -First 1
        if ($out -match '(\d+)\.') { return [int]$Matches[1] }
    } catch {}
    return 0
}

# ---------------------------------------------------------------------------
# Begin
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  AI Littering Detection — Installer" -ForegroundColor Cyan
Write-Host "  Canonical command: .\install.ps1" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# === 1. Verify Windows =====================================================
if ($PSVersionTable.Platform -and $PSVersionTable.Platform -ne 'Win32NT') {
    Step-Error "OS Check" "This installer is for Windows only (current platform: $($PSVersionTable.Platform))." -Blocking
    Stop-OnBlocking
}
Step-OK "OS Check" "Windows confirmed"

# === 2. Python >= 3.9 ======================================================
$pyVer = ""
$pyMajor = 0
$pyMinor = 0
if (Test-Command 'python') {
    try {
        $pyOut = & python --version 2>&1
        if ($pyOut -match 'Python\s+(\d+)\.(\d+)') {
            $pyMajor = [int]$Matches[1]
            $pyMinor = [int]$Matches[2]
            $pyVer = "$($Matches[1]).$($Matches[2])"
        }
    } catch {}
}
if ($pyMajor -ge 3 -and (($pyMajor -eq 3 -and $pyMinor -ge 9) -or $pyMajor -gt 3)) {
    Step-OK "Python >= 3.9" "Found Python $pyVer"
} else {
    Step-Error "Python >= 3.9" "Found '$pyVer' — install Python 3.9+ from https://python.org and re-run." -Blocking
    Stop-OnBlocking
}

# === 3. Node >= 18 =========================================================
$nodeVer = ""
$nodeMajor = 0
if (Test-Command 'node') {
    try {
        $nodeOut = & node --version 2>&1
        if ($nodeOut -match 'v(\d+)') {
            $nodeMajor = [int]$Matches[1]
            $nodeVer = $nodeOut.ToString().Trim()
        }
    } catch {}
}
if ($nodeMajor -ge 18) {
    Step-OK "Node >= 18" "Found $nodeVer"
} else {
    Step-Error "Node >= 18" "Found '$nodeVer' — install Node 18+ from https://nodejs.org and re-run." -Blocking
    Stop-OnBlocking
}

# === 4. npm ================================================================
if (Test-Command 'npm') {
    $npmVer = (& npm --version 2>&1 | Select-Object -First 1).ToString().Trim()
    Step-OK "npm" "Found npm $npmVer"
} else {
    Step-Error "npm" "npm not found — it ships with Node.js. Reinstall Node 18+." -Blocking
    Stop-OnBlocking
}

# === 5. git ================================================================
if (Test-Command 'git') {
    $gitVer = (& git --version 2>&1 | Select-Object -First 1).ToString().Trim()
    Step-OK "git" "Found $gitVer"
} else {
    Step-Error "git" "git not found — install from https://git-scm.com" -Blocking
    Stop-OnBlocking
}

# === 6. Docker (optional — WARNING if missing, not ERROR) ==================
$dockerPresent = $false
if ($SkipDocker) {
    Step-Warn "Docker Desktop" "Skipped via -SkipDocker"
} elseif (Test-Command 'docker') {
    try {
        $null = & docker info 2>&1
        if ($LASTEXITCODE -eq 0) {
            $dockerVer = (& docker --version 2>&1 | Select-Object -First 1).ToString().Trim()
            Step-OK "Docker Desktop" "$dockerVer — daemon running"
            $dockerPresent = $true
        } else {
            Step-Warn "Docker Desktop" "docker CLI found but daemon not running — start Docker Desktop"
        }
    } catch {
        Step-Warn "Docker Desktop" "docker CLI found but daemon not reachable"
    }
} else {
    Step-Warn "Docker Desktop" "Not installed — PostgreSQL will need Docker OR an external Postgres"
}

# === 7. Create Python venv (.venv) ========================================
$venvPath = Join-Path $PSScriptRoot '.venv'
if (Test-Path (Join-Path $venvPath 'Scripts\python.exe')) {
    Step-OK "Python venv" ".venv already exists"
} else {
    try {
        & python -m venv .venv 2>&1 | Out-Null
        if (Test-Path (Join-Path $venvPath 'Scripts\python.exe')) {
            Step-OK "Python venv" "Created .venv"
        } else {
            Step-Error "Python venv" "Failed to create .venv" -Blocking
            Stop-OnBlocking
        }
    } catch {
        Step-Error "Python venv" "Exception: $_" -Blocking
        Stop-OnBlocking
    }
}

# Activate venv
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
$venvPip    = Join-Path $venvPath 'Scripts\pip.exe'
if (-not (Test-Path $venvPython)) {
    Step-Error "venv activation" "python.exe not found in .venv\Scripts" -Blocking
    Stop-OnBlocking
}
Step-OK "venv activation" ".venv\Scripts\python.exe"

# === 8. pip install -r requirements.txt ===================================
Write-Host "[..] pip install -r requirements.txt (this can take a few minutes)..." -ForegroundColor DarkGray
try {
    & $venvPip install -r requirements.txt 2>&1 | ForEach-Object { Write-Host $_ -ForegroundColor DarkGray }
    if ($LASTEXITCODE -eq 0) {
        Step-OK "pip install requirements" "All packages installed"
    } else {
        Step-Error "pip install requirements" "pip exit code $LASTEXITCODE" -Blocking
        Stop-OnBlocking
    }
} catch {
    Step-Error "pip install requirements" "Exception: $_" -Blocking
    Stop-OnBlocking
}

# === 9. npm install (dashboard) ===========================================
$nodeModules = Join-Path $PSScriptRoot 'dashboard\node_modules'
if (Test-Path $nodeModules) {
    Step-OK "npm install (dashboard)" "node_modules already exists"
} else {
    Write-Host "[..] npm install (dashboard)..." -ForegroundColor DarkGray
    try {
        Push-Location (Join-Path $PSScriptRoot 'dashboard')
        & npm install 2>&1 | ForEach-Object { Write-Host $_ -ForegroundColor DarkGray }
        $npmExit = $LASTEXITCODE
        Pop-Location
        if ($npmExit -eq 0) {
            Step-OK "npm install (dashboard)" "Dependencies installed"
        } else {
            Step-Error "npm install (dashboard)" "npm exit code $npmExit"
        }
    } catch {
        Pop-Location -ErrorAction SilentlyContinue
        Step-Error "npm install (dashboard)" "Exception: $_"
    }
}

# === 10-16. Verify key Python package imports =============================
$packages = @(
    @{Name='ultralytics'; Step='10'},
    @{Name='cv2';         Step='11'},
    @{Name='fastapi';     Step='12'},
    @{Name='sqlalchemy';  Step='13'},
    @{Name='tensorflow';  Step='14'},
    @{Name='pytest';      Step='15'},
    @{Name='httpx';       Step='16'}
)
foreach ($pkg in $packages) {
    $stepName = "import $($pkg.Name)"
    try {
        & $venvPython -c "import $($pkg.Name)" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Step-OK $stepName "imported successfully"
        } else {
            Step-Error $stepName "import failed (exit $LASTEXITCODE)"
        }
    } catch {
        Step-Error $stepName "Exception: $_"
    }
}
# If any critical import failed, stop — we cannot proceed.
if ($script:ERROR_COUNT -gt 0) {
    Stop-OnBlocking
    # Even non-blocking import errors are fatal here — we need all packages
    Write-Host ""
    Write-Host "FATAL: One or more critical Python packages failed to import." -ForegroundColor Red
    Write-Host "Errors:" -ForegroundColor Red
    $script:ERROR_MESSAGES | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

# === 17. Check inference/detection/weights/best.pt ========================
$weightsPath = Join-Path $PSScriptRoot 'inference\detection\weights\best.pt'
if (Test-Path $weightsPath) {
    $size = (Get-Item $weightsPath).Length
    Step-OK "best.pt weights" "Found ($([math]::Round($size/1MB,1)) MB)"
} else {
    Step-Error "best.pt weights" @"
best.pt is MISSING — this model is REQUIRED for litter detection.

  HOW TO GET IT:
    1. Clone the reference repo: Anti-Littering-System-Computer-Vision (MIT)
       git clone https://github.com/<reference-repo>/Anti-Littering-System-Computer-Vision.git
    2. Locate best.pt inside that repo (usually in a /weights or /models folder).
    3. Copy it to:
       inference/detection/weights/best.pt

  Do NOT fabricate or substitute the file — the custom YOLO classes
  (plastic bottle, juice cup, tissue paper, ...) are baked into best.pt.
"@ -Blocking
    Write-Host ""
    Write-Host "FATAL: best.pt is required. Installation STOPPED." -ForegroundColor Red
    exit 1
}

# === 18. Check yolov8n.pt ==================================================
$yoloPath = Join-Path $PSScriptRoot 'yolov8n.pt'
if (Test-Path $yoloPath) {
    $size = (Get-Item $yoloPath).Length
    Step-OK "yolov8n.pt" "Found ($([math]::Round($size/1MB,1)) MB)"
} else {
    Step-Error "yolov8n.pt" "MISSING — download yolov8n.pt (ultralytics auto-downloads on first use, but it should be present)"
}

# === 19. Create .env from .env.example ====================================
$envFile    = Join-Path $PSScriptRoot '.env'
$envExample = Join-Path $PSScriptRoot '.env.example'
if (Test-Path $envFile) {
    Step-OK ".env" "Already exists"
} elseif (Test-Path $envExample) {
    Copy-Item $envExample $envFile
    Step-OK ".env" "Created from .env.example"
} else {
    Step-Error ".env" ".env.example not found at repo root — cannot create .env"
}

# === 20. Create evidence_store/ and inference/detection/weights/ dirs =====
$evidenceDir = Join-Path $PSScriptRoot 'evidence_store'
if (Test-Path $evidenceDir) {
    Step-OK "evidence_store/" "Directory exists"
} else {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
    Step-OK "evidence_store/" "Created"
}

$weightsDir = Join-Path $PSScriptRoot 'inference\detection\weights'
if (Test-Path $weightsDir) {
    Step-OK "inference/detection/weights/" "Directory exists"
} else {
    New-Item -ItemType Directory -Path $weightsDir -Force | Out-Null
    Step-OK "inference/detection/weights/" "Created"
}

# === 21. Docker Postgres or warn ==========================================
$dbStarted = $false
if ($dockerPresent -and -not $SkipDocker) {
    Write-Host "[..] docker-compose up -d postgres..." -ForegroundColor DarkGray
    try {
        & docker-compose up -d postgres 2>&1 | ForEach-Object { Write-Host $_ -ForegroundColor DarkGray }
        # Wait for health (up to 60 seconds)
        $healthy = $false
        for ($i = 0; $i -lt 12; $i++) {
            Start-Sleep -Seconds 5
            $status = & docker inspect --format='{{.State.Health.Status}}' littering-postgres 2>&1
            if ($status -match 'healthy') { $healthy = $true; break }
        }
        if ($healthy) {
            Step-OK "PostgreSQL" "Container healthy"
            $dbStarted = $true
        } else {
            Step-Warn "PostgreSQL" "Container started but health check did not confirm — check Docker logs"
        }
    } catch {
        Step-Warn "PostgreSQL" "docker-compose failed: $_"
    }
} else {
    Step-Warn "PostgreSQL" "Docker not available — set DATABASE_URL in .env to an external Postgres"
}

# === 22. Initialise DB schema =============================================
Write-Host "[..] Initialise DB schema..." -ForegroundColor DarkGray
# Parse DATABASE_URL from .env
$envRaw = ""
if (Test-Path $envFile) { $envRaw = Get-Content $envFile -Raw }
$dbUrl = ""
foreach ($line in ($envRaw -split "`n")) {
    if ($line -match '^DATABASE_URL\s*=\s*(.+)$') { $dbUrl = $Matches[1].Trim(); break }
}
if (-not $dbUrl) { $dbUrl = 'postgresql://litter:litter@localhost:5432/littering' }

try {
    $env:DATABASE_URL = $dbUrl
    $env:PYTHONPATH   = $PSScriptRoot
    & $venvPython -c "from backend.database import Base, create_all; create_all()" 2>&1 | ForEach-Object { Write-Host $_ -ForegroundColor DarkGray }
    if ($LASTEXITCODE -eq 0) {
        Step-OK "DB schema init" "Tables created (or already existed)"
    } else {
        Step-Warn "DB schema init" "create_all() returned exit $LASTEXITCODE — DB may be offline; tests use SQLite fallback"
    }
} catch {
    Step-Warn "DB schema init" "Exception: $_ — DB may be offline"
}

# === 23. Backend health check (start uvicorn, curl /health, kill) =========
Write-Host "[..] Backend health check..." -ForegroundColor DarkGray
$backendPort = 8000
foreach ($line in ($envRaw -split "`n")) {
    if ($line -match '^BACKEND_PORT\s*=\s*(\d+)$') { $backendPort = [int]$Matches[1]; break }
}
$healthOk = $false
$proc = $null
try {
    $env:PYTHONPATH   = $PSScriptRoot
    $env:DATABASE_URL = $dbUrl
    $proc = Start-Process -FilePath $venvPython `
        -ArgumentList @('-m', 'uvicorn', 'backend.main:app', '--port', $backendPort) `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$env:TEMP\uvicorn_out.log" `
        -RedirectStandardError  "$env:TEMP\uvicorn_err.log"

    Start-Sleep -Seconds 4

    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:$backendPort/health" -Method GET -TimeoutSec 5 -ErrorAction Stop
        if ($resp.status -eq 'ok') { $healthOk = $true }
    } catch {
        Start-Sleep -Seconds 3
        try {
            $resp = Invoke-RestMethod -Uri "http://localhost:$backendPort/health" -Method GET -TimeoutSec 5 -ErrorAction Stop
            if ($resp.status -eq 'ok') { $healthOk = $true }
        } catch {}
    }
} catch {
    # ignore — will report below
}
finally {
    if ($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}

if ($healthOk) {
    Step-OK "Backend health check" "GET /health returned status=ok"
} else {
    Step-Warn "Backend health check" "Could not reach /health — backend may need DB. Check logs: $env:TEMP\uvicorn_err.log"
}

# === 24. Frontend build ===================================================
Write-Host "[..] npm run build (dashboard)..." -ForegroundColor DarkGray
try {
    Push-Location (Join-Path $PSScriptRoot 'dashboard')
    & npm run build 2>&1 | ForEach-Object { Write-Host $_ -ForegroundColor DarkGray }
    $buildExit = $LASTEXITCODE
    Pop-Location
    $distPath = Join-Path $PSScriptRoot 'dashboard\dist'
    if ($buildExit -eq 0 -and (Test-Path $distPath)) {
        Step-OK "Frontend build" "dist/ created"
    } else {
        Step-Error "Frontend build" "npm run build exit code $buildExit"
    }
} catch {
    Pop-Location -ErrorAction SilentlyContinue
    Step-Error "Frontend build" "Exception: $_"
}

# === 25. Run tests ========================================================
Write-Host "[..] pytest tests/ -q..." -ForegroundColor DarkGray
$env:PYTHONPATH = $PSScriptRoot
try {
    & $venvPython -m pytest tests/ -q 2>&1 | ForEach-Object { Write-Host $_ -ForegroundColor DarkGray }
    $testExit = $LASTEXITCODE
    if ($testExit -eq 0) {
        Step-OK "Tests" "All tests passed"
    } else {
        Step-Error "Tests" "pytest exit code $testExit — some tests failed"
    }
} catch {
    Step-Error "Tests" "Exception: $_"
}

# === 26. AI import check ==================================================
try {
    & $venvPython -c "import ultralytics, cv2, tensorflow" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Step-OK "AI import check" "ultralytics + cv2 + tensorflow all imported"
    } else {
        Step-Error "AI import check" "import failed (exit $LASTEXITCODE)"
    }
} catch {
    Step-Error "AI import check" "Exception: $_"
}

# === 27. Smoke test: InferencePipeline ====================================
try {
    $env:PYTHONPATH = $PSScriptRoot
    & $venvPython -c "from inference.pipeline import InferencePipeline, PipelineConfig; InferencePipeline(PipelineConfig())" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Step-OK "Inference smoke test" "InferencePipeline(PipelineConfig()) instantiated"
    } else {
        Step-Error "Inference smoke test" "Pipeline instantiation failed (exit $LASTEXITCODE)"
    }
} catch {
    Step-Error "Inference smoke test" "Exception: $_"
}

# === 28. Model class validation ===========================================
if (Test-Path $weightsPath) {
    Write-Host "[..] Model class validation..." -ForegroundColor DarkGray
    $classScript = @"
import sys
sys.path.insert(0, r'$PSScriptRoot')
try:
    from ultralytics import YOLO
    model = YOLO(r'$weightsPath')
    model_classes = model.names  # dict {0: 'class_name', ...}
    class_names = sorted([str(v).lower() for v in model_classes.values()])
    print('CLASSES:' + ','.join(class_names))
except Exception as e:
    print(f'LOAD_ERROR:{type(e).__name__}:{e}')
"@
    $env:PYTHONPATH = $PSScriptRoot
    $classResult = & $venvPython -c $classScript 2>&1
    $classStr = ($classResult | Where-Object { $_ -match '^(CLASSES:|LOAD_ERROR:)' } | Select-Object -First 1).ToString().Trim()

    if ($classStr -match '^CLASSES:(.+)$') {
        $modelClassList = $Matches[1] -split ',' | ForEach-Object { $_.Trim() }
        Step-OK "Model loaded" "best.pt classes: $($modelClassList -join ', ')"

        # Compare against AssociationConfig.litter_candidate_classes
        $assocScript = @"
import sys
sys.path.insert(0, r'$PSScriptRoot')
try:
    from inference.association.person_object_assoc import AssociationConfig
    classes = list(AssociationConfig().litter_candidate_classes)
    print('ASSOC:' + ','.join(classes))
except Exception as e:
    print(f'ASSOC_ERROR:{type(e).__name__}:{e}')
"@
        $assocResult = & $venvPython -c $assocScript 2>&1
        $assocStr = ($assocResult | Where-Object { $_ -match '^(ASSOC:|ASSOC_ERROR:)' } | Select-Object -First 1).ToString().Trim()

        if ($assocStr -match '^ASSOC:(.+)$') {
            $assocClasses = $Matches[1] -split ',' | ForEach-Object { $_.Trim().ToLower() }
            # Find model classes not in assoc config
            $modelSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$modelClassList)
            $assocSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$assocClasses)
            $inModelNotAssoc = $modelSet.ExceptWith($assocSet)  # not used directly
            # Manual comparison
            $missingInAssoc = @()
            foreach ($mc in $modelClassList) {
                if ($mc -notin $assocClasses) { $missingInAssoc += $mc }
            }
            $missingInModel = @()
            foreach ($ac in $assocClasses) {
                if ($ac -notin $modelClassList) { $missingInModel += $ac }
            }
            if ($missingInAssoc.Count -eq 0 -and $missingInModel.Count -eq 0) {
                Step-OK "Class validation" "Model classes match AssociationConfig.litter_candidate_classes"
            } else {
                $detail = ""
                if ($missingInAssoc.Count -gt 0) {
                    $detail += "In model but NOT in AssociationConfig: $($missingInAssoc -join ', '). "
                }
                if ($missingInModel.Count -gt 0) {
                    $detail += "In AssociationConfig but NOT in model: $($missingInModel -join ', ')."
                }
                Step-Warn "Class validation" $detail.Trim()
            }
        } elseif ($assocStr -match '^ASSOC_ERROR:(.+)$') {
            Step-Warn "Class validation" "Could not load AssociationConfig: $($Matches[1])"
        } else {
            Step-Warn "Class validation" "Unexpected response from AssociationConfig check"
        }
    } elseif ($classStr -match '^LOAD_ERROR:(.+)$') {
        Step-Error "Model loaded" "Failed to load best.pt: $($Matches[1])"
    } else {
        Step-Warn "Model loaded" "Could not parse model classes"
    }
} else {
    Step-Warn "Model class validation" "Skipped — best.pt not found"
}

# === 29. Repository scan: fake/mock/placeholder/dummy in production code ===
Write-Host "[..] Repository scan for fake/mock/placeholder/dummy..." -ForegroundColor DarkGray
$scanPatterns = @('fake', 'mock', 'placeholder', 'dummy')
$excludeDirs = @('node_modules', '.venv', '__pycache__', '.git', 'tests', '.pytest_cache')
$excludeExts = @('.pyc', '.log')
$findings = @()

$prodFiles = Get-ChildItem -Path $PSScriptRoot -Recurse -File `
    | Where-Object {
        $rel = $_.FullName.Substring($PSScriptRoot.Length)
        $excluded = $false
        foreach ($d in $excludeDirs) { if ($rel -match "\\$d\\") { $excluded = $true; break } }
        foreach ($e in $excludeExts) { if ($_.Extension -eq $e) { $excluded = $true; break } }
        -not $excluded -and ($_.Extension -eq '.py' -or $_.Extension -eq '.ts' -or $_.Extension -eq '.tsx' -or $_.Extension -eq '.js')
    }

foreach ($file in $prodFiles) {
    foreach ($pattern in $scanPatterns) {
        $matches = Select-String -Path $file.FullName -Pattern $pattern -AllMatches -ErrorAction SilentlyContinue
        foreach ($m in $matches) {
            $findings += [PSCustomObject]@{
                File    = $file.FullName.Substring($PSScriptRoot.Length + 1)
                Line    = $m.LineNumber
                Pattern = $pattern
                Text    = $m.Line.Trim()
            }
        }
    }
}

if ($findings.Count -eq 0) {
    Step-OK "Repository scan" "No fake/mock/placeholder/dummy found in production code"
} else {
    $detail = "$($findings.Count) occurrence(s) found in production code:"
    Step-Warn "Repository scan" $detail
    $findings | Select-Object -First 20 | ForEach-Object {
        Write-Host "      $($_.File):$($_.Line) [$($_.pattern)] $($_.Text)" -ForegroundColor DarkYellow
    }
    if ($findings.Count -gt 20) {
        Write-Host "      ... and $($findings.Count - 20) more" -ForegroundColor DarkYellow
    }
}

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  INSTALLATION REPORT" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  OK       : $script:OK_COUNT" -ForegroundColor Green
Write-Host "  Warnings : $script:WARN_COUNT" -ForegroundColor Yellow
Write-Host "  Errors   : $script:ERROR_COUNT" -ForegroundColor Red
Write-Host "  Blocking : $($script:BLOCKING.Count)" -ForegroundColor Red
Write-Host "==========================================" -ForegroundColor Cyan

if ($script:ERROR_COUNT -gt 0) {
    Write-Host ""
    Write-Host "  BLOCKING ISSUES:" -ForegroundColor Red
    $script:BLOCKING | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    if ($script:ERROR_MESSAGES.Count -gt $script:BLOCKING.Count) {
        Write-Host ""
        Write-Host "  OTHER ERRORS:" -ForegroundColor Red
        $script:ERROR_MESSAGES | Where-Object { $_ -notin $script:BLOCKING } | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    }
}

Write-Host ""
if ($script:ERROR_COUNT -eq 0) {
    Write-Host "  Installation PASS" -ForegroundColor Green
    Write-Host "  Next action: run .\start.ps1 to launch the system." -ForegroundColor Green
    exit 0
} else {
    Write-Host "  Installation FAIL" -ForegroundColor Red
    Write-Host "  Next action: fix the blocking issues above, then re-run: .\install.ps1" -ForegroundColor Red
    exit 1
}
