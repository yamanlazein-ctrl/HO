# Contribution & Originality

## What we took from the reference project

From `Anti-Littering-System-Computer-Vision` (MIT licensed):

- **`best.pt`** — the YOLO garbage-detection weights (plastic bottle,
  juice cup, tissue paper, ...).
- **MoveNet integration idea** — using body keypoints to reason about
  hand proximity to objects.
- **Initial hand–object distance logic** — as a reference, then
  substantially redesigned.
- **DeepFace integration** — kept as an *optional, future* component.

## What we did NOT take

- The simple single-frame decision logic.
- The Excel-based fine tracking.
- The script-style architecture (no modules, no tests, no backend).

## Our original contributions (the 🔴 layers)

### 1. Person–Object Association (`inference/association/`)
A sticky, temporal binding between a person track and an object track
with:
- persistence gating (a pair must co-occur for N frames before
  established),
- ID-switch rebind (rebounds to a new object track within
  `reassoc_window` when ByteTrack loses the bottle mid-throw),
- hand-occlusion fallback (torso-center distance when wrist keypoints
  are missing at release).

### 2. Temporal Littering State Machine (`inference/behavior/state_machine.py`)
A per-pair FSM whose key novelty is **explicit reversion edges**:
- RELEASE → HOLDING (re-grasp within `regrasp_window`)
- OBJECT_ON_GROUND → HOLDING (object retrieved)
- OBJECT_ON_GROUND → NORMAL (put-down, person never leaves within
  `abandon_window`)
- PERSON_AWAY → OBJECT_ON_GROUND (person came back)
- SUSPICIOUS → NORMAL (suspicion decay)
- any → NORMAL (both tracks lost)

This is what separates "littering" from "interacting with an object".

### 3. Temporal Voting (`inference/behavior/voting.py`)
Multi-observation weighted scoring with decay. Re-grasp and
person-returned are strong negatives (-4.0) that flip a borderline
CONFIRM into a REVERT. The decision is never made from a single frame.

### 4. Evidence Generation (`inference/evidence/` + `inference/capture/circular_buffer.py`)
A time-based circular buffer that retains the last N seconds
continuously, so the evidence clip is pre-event + event + post-event
without ever starting recording late. The post-window finalize step is
explicit, which explains (and owns) the ~3s demo latency.

### 5. End-to-End Integration + Evaluation Methodology
The pipeline orchestration, the FastAPI backend, the PostgreSQL schema,
the React dashboard, and the **per-scenario evaluation framework** that
proves behavioral discrimination rather than a single F1 number.

## How to defend this to the committee

The thesis contribution is **not** "we ran YOLO on litter". It is:
> a temporal behavior analysis layer that confirms a littering event
> from a multi-observation sequence, with explicit reversion paths that
> prevent false positives on look-alike behaviors (put-down, re-grasp,
> accidental drop + retrieval).

The unit tests (`tests/`) demonstrate the contribution on pure logic —
no camera, no CV models — which is the strongest possible evidence that
the contribution is real and not an artifact of model luck.

## Scientifically honest terminology (research-backed)

Research confirms that human intent is an internal, indirectly observable
state — the same physical action can stem from different motives, so there
is no one-to-one mapping between observable motion and intent [1][2]. A
single fixed camera observing only visible behavior cannot reliably
distinguish "intentional littering" from "accidental dropping."

**Therefore the system detects a "littering event candidate" — a
behavioral sequence matching the littering signature — NOT a definitive
legal violation.** Every confirmed candidate is surfaced for human
review in the dashboard. The committee should understand the system
as a detection-and-evidence tool that flags candidates, not a legal
adjudicator.

This aligns with the ETRI 2019 framing: the system "detects the
dumping action by the change in relation between a person and the
object" — it detects the action pattern, not the dumper's intent [3].

[1] Intent recognition literature: intent is an internal state,
    indirectly observable; same action → different motives.
[2] ExaAnswer research confirmation (2026): "Computer vision cannot
    currently distinguish intentional littering from accidental dropping
    with high reliability using only observable behavior from a single
    fixed camera."
[3] Yun et al., "Vision-based garbage dumping action detection for
    real-world surveillance platform," ETRI Journal, 2019
    (DOI: 10.4218/etrij.2018-0520).
