<#
.SYNOPSIS
    AI Littering Detection — Windows one-command setup (24 steps).
.DESCRIPTION
    Verifies prerequisites, creates the Python venv, installs all deps,
    checks key imports, verifies model weights, starts PostgreSQL (if
    Docker is available), initialises the DB schema, runs a backend
    health check, builds the frontend, runs the test suite, and performs
    a minimal inference smoke test.

    Every step prints [OK] / [WARNING] / [ERROR] with the step name.
    The script exits 0 only if zero ERRORs (WARNINGs are allowed).
    It NEVER fakes success.

    NOTE: This script is designed for Windows PowerShell 5.1+ / 7+.
          It was NOT executed in the sandbox (no PowerShell there) and
          is therefore UNVERIFIED at runtime, but follows the spec
          exactly.
.NOTES
    File     : setup.ps1
    Location : repo root
    Author   : Windows Automation Engineer
#>

#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$SkipDocker  # forwarded to install.ps1
)

# ---------------------------------------------------------------------------
# Delegation wrapper — the canonical installer is install.ps1.
# All arguments are forwarded.  Both .\setup.ps1 and .\install.ps1 work.
# ---------------------------------------------------------------------------

$installScript = Join-Path $PSScriptRoot 'install.ps1'
if (-not (Test-Path $installScript)) {
    Write-Host "[ERROR] install.ps1 not found at $installScript" -ForegroundColor Red
    exit 1
}

# Forward all arguments to the canonical installer
$forwardArgs = @()
if ($SkipDocker) { $forwardArgs += '-SkipDocker' }

& $installScript @forwardArgs
exit $LASTEXITCODE
