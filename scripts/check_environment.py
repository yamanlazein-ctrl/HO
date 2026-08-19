#!/usr/bin/env python3
"""
AI Littering Detection — Cross-Platform Environment Validator.

Checks and reports [OK] / [WARNING] / [ERROR] / [WAITING] for every
component the system depends on.  This script is designed to run on
Windows, macOS, and Linux — it uses no Windows-only APIs and guards
every platform-specific call with ``sys.platform`` checks.

Exit codes:
    0  — no [ERROR] (warnings / waiting are acceptable)
    1  — at least one [ERROR]

Usage:
    PYTHONPATH=. python scripts/check_environment.py
    PYTHONPATH=. python scripts/check_environment.py --verbose

Author: Windows Automation Engineer
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import tempfile
import importlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env if present (simple parser — no python-dotenv dependency required)
def _load_dotenv() -> dict[str, str]:
    env: dict[str, str] = {}
    dotenv_path = _REPO_ROOT / ".env"
    if not dotenv_path.exists():
        dotenv_path = _REPO_ROOT / ".env.example"
    if dotenv_path.exists():
        for line in dotenv_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip()
    return env

_DOTENV = _load_dotenv()


def _env(key: str, default: str = "") -> str:
    """Return value from os.environ first, then .env, then default."""
    return os.environ.get(key, _DOTENV.get(key, default))


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    status: str          # "OK" | "WARNING" | "ERROR" | "WAITING"
    name: str
    detail: str = ""

@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.results if r.status == "OK")

    @property
    def warn_count(self) -> int:
        return sum(1 for r in self.results if r.status == "WARNING")

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.status == "ERROR")

    @property
    def waiting_count(self) -> int:
        return sum(1 for r in self.results if r.status == "WAITING")

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.results.append(CheckResult(status, name, detail))
        colour = {
            "OK":      "\033[92m",  # green
            "WARNING": "\033[93m",  # yellow
            "ERROR":   "\033[91m",  # red
            "WAITING": "\033[96m",  # cyan
        }.get(status, "")
        reset = "\033[0m"
        line = f"  [{status}] {name}"
        if detail:
            line += f" — {detail}"
        print(f"{colour}{line}{reset}")

    def has_errors(self) -> bool:
        return self.error_count > 0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _run_cmd(cmd: list[str], timeout: int = 10) -> Tuple[int, str, str]:
    """Run a subprocess, return (exit_code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as e:
        return 1, "", str(e)


def _try_import(module_name: str) -> Tuple[bool, str]:
    """Attempt to import a module. Returns (success, version_or_error)."""
    try:
        mod = importlib.import_module(module_name)
        ver = getattr(mod, "__version__", "")
        return True, ver
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_python_version(report: Report) -> None:
    """Check Python >= 3.9."""
    vi = sys.version_info
    ver_str = f"{vi.major}.{vi.minor}.{vi.micro}"
    if vi.major >= 3 and vi.minor >= 9:
        report.add("OK", "Python version", f"{ver_str} (>= 3.9)")
    elif vi.major > 3:
        report.add("OK", "Python version", f"{ver_str} (>= 3.9)")
    else:
        report.add("ERROR", "Python version", f"{ver_str} — need >= 3.9")


def check_node_npm(report: Report) -> None:
    """Check Node >= 18 and npm."""
    node_path = shutil.which("node")
    if node_path:
        code, out, _ = _run_cmd(["node", "--version"])
        if code == 0 and out.startswith("v"):
            major = int(out[1:].split(".")[0])
            if major >= 18:
                report.add("OK", "Node.js", f"{out} (>= 18)")
            else:
                report.add("ERROR", "Node.js", f"{out} — need >= 18")
        else:
            report.add("ERROR", "Node.js", f"unexpected output: {out}")
    else:
        report.add("WARNING", "Node.js", "not on PATH — dashboard build will be skipped")

    npm_path = shutil.which("npm")
    if npm_path:
        code, out, _ = _run_cmd(["npm", "--version"])
        if code == 0:
            report.add("OK", "npm", f"v{out}")
        else:
            report.add("ERROR", "npm", f"npm --version failed (exit {code})")
    else:
        report.add("WARNING", "npm", "not on PATH")


def check_git(report: Report) -> None:
    git_path = shutil.which("git")
    if git_path:
        code, out, _ = _run_cmd(["git", "--version"])
        if code == 0:
            report.add("OK", "git", out)
        else:
            report.add("WARNING", "git", "installed but --version failed")
    else:
        report.add("WARNING", "git", "not on PATH")


def check_docker(report: Report) -> None:
    docker_path = shutil.which("docker")
    if docker_path:
        code, out, _ = _run_cmd(["docker", "--version"])
        if code == 0:
            report.add("OK", "Docker", out)
        else:
            report.add("WARNING", "Docker", "CLI found but --version failed")
    else:
        report.add("WARNING", "Docker",
                    "not installed — PostgreSQL needs Docker or an external Postgres")


def check_python_packages(report: Report) -> None:
    """Check required Python packages by import attempt."""
    packages = [
        ("ultralytics", "ultralytics"),
        ("cv2",         "opencv"),
        ("fastapi",     "fastapi"),
        ("sqlalchemy",  "sqlalchemy"),
        ("tensorflow",  "tensorflow"),
        ("pytest",      "pytest"),
        ("httpx",       "httpx"),
    ]
    for import_name, display_name in packages:
        ok, info = _try_import(import_name)
        if ok:
            ver = f" v{info}" if info else ""
            report.add("OK", f"Package: {display_name}", f"imported{ver}")
        else:
            report.add("ERROR", f"Package: {display_name}", info)


def check_model_files(report: Report) -> None:
    """Check best.pt exists; check yolov8n.pt exists or is downloadable."""
    # best.pt
    best_pt = _REPO_ROOT / "inference" / "detection" / "weights" / "best.pt"
    if best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        report.add("OK", "Model: best.pt", f"found ({size_mb:.1f} MB)")
    else:
        report.add("ERROR", "Model: best.pt",
                    "MISSING — download from the reference repo "
                    "(Anti-Littering-System-Computer-Vision, MIT) and place at "
                    "inference/detection/weights/best.pt")

    # yolov8n.pt
    yolov8n = _REPO_ROOT / "yolov8n.pt"
    if yolov8n.exists():
        size_mb = yolov8n.stat().st_size / (1024 * 1024)
        report.add("OK", "Model: yolov8n.pt", f"found ({size_mb:.1f} MB)")
    else:
        # Check if ultralytics can auto-download it
        try:
            from ultralytics import YOLO  # noqa: F401
            report.add("WARNING", "Model: yolov8n.pt",
                        "not in repo root — ultralytics will auto-download on first use")
        except Exception:
            report.add("WARNING", "Model: yolov8n.pt",
                        "not found and ultralytics not installed — will need manual download")


def check_opencv_ultralytics_tensorflow(report: Report) -> None:
    """Deeper check: OpenCV video capture, Ultralytics YOLO, TensorFlow/MoveNet."""
    # OpenCV
    ok, info = _try_import("cv2")
    if ok:
        try:
            import cv2
            ver = cv2.__version__
            report.add("OK", "OpenCV", f"v{ver}")
        except Exception as e:
            report.add("ERROR", "OpenCV", f"import succeeded but error: {e}")
    else:
        report.add("ERROR", "OpenCV", info)

    # Ultralytics
    ok, info = _try_import("ultralytics")
    if ok:
        report.add("OK", "Ultralytics (YOLO)", f"v{info}" if info else "imported")
    else:
        report.add("ERROR", "Ultralytics (YOLO)", info)

    # TensorFlow / MoveNet
    ok, info = _try_import("tensorflow")
    if ok:
        try:
            import tensorflow as tf
            report.add("OK", "TensorFlow (MoveNet)", f"v{tf.__version__}")
        except Exception as e:
            report.add("ERROR", "TensorFlow (MoveNet)", f"import error: {e}")
    else:
        report.add("ERROR", "TensorFlow (MoveNet)", info)


def check_database(report: Report) -> None:
    """Try connecting to DATABASE_URL. Failure is [WARNING], not [ERROR]."""
    db_url = _env("DATABASE_URL", "postgresql://litter:litter@localhost:5432/littering")
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        report.add("OK", "Database", f"connected to {db_url.split('@')[-1] if '@' in db_url else db_url}")
    except Exception as e:
        report.add("WARNING", "Database",
                    f"cannot connect ({type(e).__name__}) — DB can be external/optional for tests. "
                    f"URL: {db_url}")


def check_backend_endpoints(report: Report) -> None:
    """If backend is running, GET /health and /api/status."""
    backend_port = _env("BACKEND_PORT", "8000")
    base_url = f"http://localhost:{backend_port}"

    # /health
    try:
        import httpx
        resp = httpx.get(f"{base_url}/health", timeout=5)
        if resp.status_code == 200 and resp.json().get("status") == "ok":
            report.add("OK", "Backend: /health", f"200 OK (port {backend_port})")
        else:
            report.add("WARNING", "Backend: /health", f"status {resp.status_code}")
    except httpx.ConnectError:
        report.add("WAITING", "Backend: /health",
                    f"not running on port {backend_port} — start with uvicorn backend.main:app")
    except Exception as e:
        report.add("WARNING", "Backend: /health", f"{type(e).__name__}: {e}")

    # /api/status
    try:
        import httpx
        resp = httpx.get(f"{base_url}/api/status", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            ai_status = data.get("ai_engine", {}).get("status", "unknown")
            cam_status = data.get("camera", {}).get("status", "unknown")
            report.add("OK", "Backend: /api/status",
                        f"200 (ai_engine={ai_status}, camera={cam_status})")
        else:
            report.add("WARNING", "Backend: /api/status", f"status {resp.status_code}")
    except Exception:
        # If /health already said not running, don't double-report
        pass


def check_frontend_build(report: Report) -> None:
    """Check if dashboard/dist exists (production build)."""
    dist_path = _REPO_ROOT / "dashboard" / "dist"
    if dist_path.exists() and any(dist_path.iterdir()):
        files = list(dist_path.rglob("*"))
        report.add("OK", "Frontend build (dist/)", f"{len(files)} files in dashboard/dist/")
    else:
        report.add("WARNING", "Frontend build (dist/)",
                    "not built — run: cd dashboard && npm run build")


def check_evidence_directory(report: Report) -> None:
    """Check evidence_store/ exists and is writable."""
    ev_dir = _REPO_ROOT / "evidence_store"
    if not ev_dir.exists():
        try:
            ev_dir.mkdir(parents=True, exist_ok=True)
            report.add("OK", "Evidence directory", "created evidence_store/")
        except Exception as e:
            report.add("ERROR", "Evidence directory", f"cannot create: {e}")
            return
    else:
        report.add("OK", "Evidence directory", "exists")

    # Write-permission test
    test_file = ev_dir / f".write_test_{os.getpid()}.tmp"
    try:
        test_file.write_text("ok")
        test_file.unlink()
        report.add("OK", "Evidence write permission", "writable")
    except Exception as e:
        report.add("ERROR", "Evidence write permission", f"cannot write: {e}")


def check_camera(report: Report) -> None:
    """
    Try cv2.VideoCapture(0).
    If it opens, [OK]. Otherwise [WAITING FOR CAMERA] — NEVER claim [OK]
    unless the camera actually opens.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap is not None and cap.isOpened():
            # Read one frame to confirm it's real
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                report.add("OK", "Camera (device 0)", f"opened, frame {w}x{h}")
            else:
                report.add("WAITING", "Camera (device 0)",
                            "opened but could not read frame — check camera connection")
        else:
            if cap is not None:
                cap.release()
            report.add("WAITING", "Camera (device 0)",
                        "WAITING FOR CAMERA — cv2.VideoCapture(0) did not open. "
                        "Connect a webcam or iPhone (via Camo/Iriun) and re-run.")
    except ImportError:
        report.add("ERROR", "Camera (device 0)", "OpenCV (cv2) not installed — cannot check camera")
    except Exception as e:
        report.add("WAITING", "Camera (device 0)",
                    f"cv2.VideoCapture(0) failed: {type(e).__name__}: {e}")


def check_inference_pipeline(report: Report) -> None:
    """Smoke test: instantiate InferencePipeline(PipelineConfig())."""
    try:
        from inference.pipeline import InferencePipeline, PipelineConfig
        pipeline = InferencePipeline(PipelineConfig())
        report.add("OK", "Inference pipeline", "InferencePipeline(PipelineConfig()) instantiated")
    except Exception as e:
        report.add("ERROR", "Inference pipeline",
                    f"failed to instantiate: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print()
    print("=" * 60)
    print("  AI Littering Detection — Environment Check")
    print(f"  Platform: {sys.platform}  |  Python: {sys.version.split()[0]}")
    print(f"  Repo:     {_REPO_ROOT}")
    print(f"  Time:     {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60)
    print()

    report = Report()

    # --- Core runtime ---
    print("── Core Runtime ──")
    check_python_version(report)
    check_node_npm(report)
    check_git(report)
    check_docker(report)
    print()

    # --- Python packages ---
    print("── Python Packages ──")
    check_python_packages(report)
    print()

    # --- AI / CV libraries ---
    print("── AI / Computer Vision ──")
    check_opencv_ultralytics_tensorflow(report)
    print()

    # --- Model files ---
    print("── Model Files ──")
    check_model_files(report)
    print()

    # --- Inference pipeline ---
    print("── Inference Pipeline ──")
    check_inference_pipeline(report)
    print()

    # --- Database ---
    print("── Database ──")
    check_database(report)
    print()

    # --- Backend endpoints ---
    print("── Backend Endpoints ──")
    check_backend_endpoints(report)
    print()

    # --- Frontend ---
    print("── Frontend ──")
    check_frontend_build(report)
    print()

    # --- Evidence + permissions ---
    print("── Evidence Store ──")
    check_evidence_directory(report)
    print()

    # --- Camera ---
    print("── Camera ──")
    check_camera(report)
    print()

    # --- Summary ---
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  OK:       {report.ok_count}")
    print(f"  WARNING:  {report.warn_count}")
    print(f"  WAITING:  {report.waiting_count}")
    print(f"  ERROR:    {report.error_count}")
    print("=" * 60)

    if report.error_count > 0:
        print()
        print("  ERRORS:")
        for r in report.results:
            if r.status == "ERROR":
                print(f"    • {r.name}: {r.detail}")
        print()
        print("  ❌ Environment check FAILED — resolve errors above.")
        return 1

    if report.waiting_count > 0:
        print()
        print("  ⏳ Some components are WAITING (camera / backend). "
              "These are not errors — start them when ready.")

    if report.warn_count > 0:
        print()
        print("  ⚠️  Warnings are non-blocking but should be reviewed.")

    print()
    print("  ✅ Environment check PASSED — no errors detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
