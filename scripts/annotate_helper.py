"""Contact sheets to annotate dev-clip event start/end quickly (plan.md W1 task).

  python scripts/annotate_helper.py            # writes report/annotate/<clip>.jpg

Each sheet tiles one thumbnail every 0.5 s with its timestamp. Read off the first and last
frames of the incident and fill event_start_s / event_end_s in data/annotations_devset.csv
(several segments: separate with ';'). We do not invent these times automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.config import ROOT
from xstream.datasets import devset

STEP, W, COLS = 0.5, 240, 8


def sheet(path: Path, out: Path, max_s: float = 90) -> None:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tiles, t = [], 0.0
    while t < max_s:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (W, int(f.shape[0] * W / f.shape[1])))
        cv2.putText(f, f"{t:.1f}s", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(f, f"{t:.1f}s", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(f)
        t += STEP
    cap.release()
    if not tiles:
        return
    h = tiles[0].shape[0]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % COLS)
    rows = [np.hstack([cv2.resize(x, (W, h)) for x in tiles[i:i + COLS]]) for i in range(0, len(tiles), COLS)]
    cv2.imwrite(str(out), np.vstack(rows))


if __name__ == "__main__":
    out = ROOT / "report" / "annotate"
    out.mkdir(parents=True, exist_ok=True)
    for v in devset():
        if v.path and v.label != "none":
            sheet(v.path, out / f"{v.source_id}.jpg")
            print(out / f"{v.source_id}.jpg")
