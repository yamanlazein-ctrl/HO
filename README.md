# AI-Based CCTV Littering Detection and Evidence System

---

## Windows Quick Start

Two commands set up and launch the entire system on Windows (PowerShell 5.1+ or 7+):

```powershell
# 1) Install everything — venv, Python deps, npm, Postgres, DB schema, tests
.\setup.ps1

# 2) Start backend + dashboard + Postgres in background windows
.\start.ps1
```

**What `.\setup.ps1` does (24 steps):** verifies Windows / Python ≥ 3.9 / Node ≥ 18 / npm / git / Docker, creates `.venv`, `pip install -r requirements.txt`, `npm install` the dashboard, verifies `ultralytics` / `cv2` / `fastapi` / `sqlalchemy` / `tensorflow` imports, checks `best.pt` exists, creates `.env` + `evidence_store/`, starts PostgreSQL (if Docker), initialises DB schema, runs a backend `/health` check, builds the frontend, runs `pytest`, and does an inference-pipeline smoke test. Every step prints `[OK]` / `[WARNING]` / `[ERROR]`. Exit 0 only if zero errors.

**What `.\start.ps1` does:** verifies installation, starts PostgreSQL (if Docker), launches `uvicorn backend.main:app` and the Vite dev server in separate windows, then prints the status summary block.

### Prerequisites

| Tool | Minimum | Notes |
|------|---------|-------|
| Python | 3.9 | https://python.org — check "Add to PATH" on install |
| Node.js | 18 | https://nodejs.org |
| Git | any | https://git-scm.com |
| Docker Desktop | optional | needed for local PostgreSQL; can use external Postgres instead |
| Webcam / iPhone | optional | Camo or Iriun for iPhone-as-camera; needed for live detection |

### Getting `best.pt` (required for litter detection)

`best.pt` is the custom YOLO model trained on litter classes (plastic bottle, juice cup, tissue paper, …). It is **not** included in this repo.

1. Clone the reference repo: `Anti-Littering-System-Computer-Vision` (MIT license).
2. Locate `best.pt` inside that repo (usually in a `/weights` or `/models` folder).
3. Copy it to: `inference/detection/weights/best.pt`

Setup will `[ERROR]` and stop if this file is missing.

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `python` not found | Reinstall Python 3.9+ and tick "Add Python to PATH" |
| `pip install` fails (ultralytics/tensorflow) | Ensure Python 3.9–3.12; TF does not support 3.13+ yet. Try `pip install --upgrade pip` first |
| `best.pt MISSING` | Download from the reference repo — see "Getting best.pt" above |
| Docker not running | Start Docker Desktop, then re-run `.\setup.ps1`. Or skip with `.\setup.ps1 -SkipDocker` and set `DATABASE_URL` in `.env` to an external Postgres |
| PostgreSQL health check fails | `docker logs littering-postgres` — wait 30s and retry |
| Backend `/health` not responding | Check the uvicorn window for errors; ensure port 8000 is free (`netstat -ano \| findstr :8000`) |
| `npm run build` fails | `cd dashboard && rm -r node_modules && npm install` then retry |
| Camera `[WAITING FOR CAMERA]` | Connect a webcam or iPhone (Camo/Iriun). On Windows: Settings → Privacy → Allow camera access. Try `CAMERA_DEVICE_INDEX=1` in `.env` |
| `yolov8n.pt` not found | Ultralytics auto-downloads it on first inference — no manual action needed |
| Tests fail | `PYTHONPATH=. .venv\Scripts\python.exe -m pytest tests/ -v` for details |
| Port already in use | Change `BACKEND_PORT` / `DASHBOARD_PORT` in `.env` |

### Diagnostic scripts

```powershell
# Deep diagnostic — checks everything and reports [OK]/[WARNING]/[ERROR]/[WAITING]
.\scripts\doctor.ps1

# Cross-platform Python validator (also works on Linux/macOS)
PYTHONPATH=. .venv\Scripts\python.exe scripts\check_environment.py
```

---

نظام يستخدم كاميرا iPhone ثابتة لمراقبة المشهد، ويستخدم نماذج جاهزة (YOLO, ByteTrack, MoveNet) لاكتشاف الأشخاص والأجسام وتتبعها، ثم يضيف **طبقة سلوكية زمنية طورناها** لتحليل العلاقة بين الشخص والنفاية عبر الزمن، تأكيد حدث الرمي، وحفظ دليل مرئي، ثم عرضه عبر Backend (FastAPI) وDashboard (React).

> **Scope:** النسخة الأولى تفترض كاميرا ثابتة (Static Camera) وإضاءة نهارية كافية. معالجة حركة الكاميرا والتعرف على الوجه Future Work.

---

## أين مساهمتنا؟

الطبقات 🔴 هي مساهمة المشروع الأساسية (كود حقيقي قابل للاختبار وحدة بدون كاميرا):

| الطبقة | المكوّن | الحالة |
|---|---|---|
| Video | iPhone + OpenCV | 🟢 جاهز |
| Detection | YOLO (`best.pt`) | 🟢 جاهز |
| Tracking | ByteTrack | 🟢 جاهز |
| Pose | MoveNet | 🟢 جاهز |
| **Association** | **Person–Object عبر الزمن** | **🔴 نبنيه** |
| **Behavior** | **Temporal State Machine** | **🔴 نبنيه** |
| **Decision** | **Temporal Voting** | **🔴 نبنيه** |
| **Evidence** | **Circular Buffer + Recorder** | **🔴 نبنيه** |
| Backend | FastAPI | 🔵 نطبّق framework |
| Database | PostgreSQL | 🔵 نطبّق framework |
| Dashboard | React + Tailwind + shadcn | 🔵 نطبّق framework |
| Face | DeepFace | 🟡 اختياري |
| Evaluation | Metrics + Test Dataset | 🔴 نبنيه |

---

## البنية

```
inference/            # محرك الذكاء الاصطناعي
  capture/            # camera_source + circular_buffer 🔴
  detection/          # YOLO 🟢
  tracking/           # ByteTrack 🟢
  pose/               # MoveNet 🟢
  association/        # person_object_assoc 🔴
  behavior/           # state_machine 🔴 + voting 🔴
  evidence/           # evidence_manager 🔴
backend/              # FastAPI 🔵
dashboard/            # React 🔵
evaluation/           # metrics 🔴 + dataset schema
tests/                # unit tests للطبقات 🔴
scripts/              # run_pipeline + evaluate
docs/                 # architecture, setup, contribution
```

## التشغيل السريع

```bash
# 1) رفع PostgreSQL
docker-compose up -d postgres

# 2) تثبيت اعتماديات المحرك
pip install -r requirements.txt

# 3) اختبارات الوحدة (تعمل بلا كاميرا — منطق نقي)
pytest tests/ -v

# 4) تشغيل المحرك (على لابتوبك مع الكاميرا)
python scripts/run_pipeline.py --source camo --buffer 6

# 5) تشغيل الـBackend
uvicorn backend.main:app --reload

# 6) تشغيل الـDashboard
cd dashboard && npm install && npm run dev
```

راجع `docs/SETUP.md` لتفاصيل نقل فيديو الـiPhone عبر Camo/Iriun.

## التقييم

نصنع test dataset صغير (100 clip: 50 littering + 50 normal) ونقيس per-scenario:
Event Precision, Event Recall, F1, False Positive Rate, FPS.

راجع `evaluation/dataset_schema.md` و `docs/ARCHITECTURE.md`.

## المصدر المرجعي

من `Anti-Littering-System-Computer-Vision` (MIT) نأخذ: `best.pt`، فكرة دمج MoveNet، منطق مسافة اليد-الجسم. الباقي بنية جديدة طورناها.
