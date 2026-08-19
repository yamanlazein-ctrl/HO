# Test Dataset Schema

The evaluation dataset is a **controlled behavioral test set** — small,
purpose-built, and clearly labeled as such in the thesis (not a claim of
in-the-wild performance). Its job is to prove the system distinguishes
*behaviors*, not just detects trash.

## Size

- 100 video clips total
- 50 littering (positive)
- 50 normal (negative)

## Scenarios

### Littering (positive) — 50 clips

| Scenario | Count | Description |
|---|---|---|
| `throw_bottle` | 15 | Person throws a plastic bottle, walks away |
| `throw_cup` | 10 | Person throws a cup/can, walks away |
| `drop_object` | 10 | Person drops object (accidental-looking), walks away |
| `leave_object` | 10 | Person places object down and leaves it |
| `toss_paper` | 5 | Person tosses tissue/wrapper, walks away |

### Normal (negative) — 50 clips

| Scenario | Count | Description |
|---|---|---|
| `carry_bottle` | 15 | Person carries a bottle across the frame |
| `put_down_pick_up` | 15 | Person puts bottle down, then picks it back up |
| `drink_from_cup` | 10 | Person drinks from a cup, carries it away |
| `walk_near_trash` | 5 | Person walks past existing trash on ground |
| `accidental_drop_retrieval` | 5 | Person drops object, picks it up immediately |

The `put_down_pick_up` and `accidental_drop_retrieval` scenarios are the
**critical false-positive guards** — they exercise the FSM's reversion
paths (RELEASE → HOLDING via re-grasp; OBJECT_ON_GROUND → NORMAL via
abandon_window). If the system confirms on these, the reversion logic is
broken.

## Clip format

```
dataset/
  littering/
    throw_bottle_001.mp4
    throw_bottle_001.json    # ground truth annotation
    ...
  normal/
    carry_bottle_001.mp4
    carry_bottle_001.json
    ...
```

### Annotation JSON

```json
{
  "clip_id": "throw_bottle_001",
  "scenario": "throw_bottle",
  "ground_truth": true,
  "event_timestamp_sec": 3.2,
  "object_type": "plastic bottle",
  "notes": "right-hand throw, bottle lands center-frame"
}
```

## Filming protocol (for the team)

1. **Static iPhone** on a tripod, Camo/Iriun running, 1080p, 30fps.
2. **Daytime, adequate lighting** (scope constraint).
3. **One actor** per clip (avoid multi-person ambiguity in the MVP).
4. **Same background** for littering and normal clips of each object
   type — isolates the behavior variable.
5. Clip length: 5–8 seconds each (covers pre/event/post).
6. Label the annotation JSON immediately after filming.

## What we measure

See `evaluation/metrics.py`. The per-scenario table is the defensible
artifact presented to the committee — it shows the *behavioral
discrimination*, not just a single F1.
