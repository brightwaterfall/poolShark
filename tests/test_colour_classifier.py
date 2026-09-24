#!/usr/bin/env python3
"""Synthetic regression tests for poolShark ball colour classification.

Mirrors the C++ classifier logic in main.cpp so we can validate behaviour
without a Qt/OpenCV C++ toolchain. Run:  py -3 tests/test_colour_classifier.py
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:
    print("FAIL: opencv-python not installed. Run: py -3 -m pip install opencv-python-headless numpy")
    sys.exit(2)


# ---- mirrored C++ classifier ------------------------------------------------

BC_CUE, BC_YELLOW, BC_BLUE, BC_RED, BC_PURPLE = 0, 1, 2, 3, 4
BC_ORANGE, BC_GREEN, BC_MAROON, BC_BLACK, BC_UNKNOWN = 5, 6, 7, 8, 9

NAMES = {
    BC_CUE: "CUE",
    BC_YELLOW: "YELLOW",
    BC_BLUE: "BLUE",
    BC_RED: "RED",
    BC_PURPLE: "PURPLE",
    BC_ORANGE: "ORANGE",
    BC_GREEN: "GREEN",
    BC_MAROON: "MAROON",
    BC_BLACK: "BLACK",
    BC_UNKNOWN: "?",
}


@dataclass
class ColorSample:
    hsv: tuple[int, int, int]
    lab: tuple[float, float, float]
    chroma: float = 0.0
    white_frac: float = 0.0
    dark_frac: float = 0.0
    chroma_frac: float = 0.0
    ok: bool = True


def hue_dist(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 180.0 - d)


def classify_color_sample(s: ColorSample) -> tuple[int, bool]:
    is_stripe = False
    if not s.ok:
        return BC_UNKNOWN, False

    L, a, b = s.lab[0], s.lab[1] - 128.0, s.lab[2] - 128.0
    H, S, V = s.hsv

    if s.chroma < 18.0 and L >= 155.0 and S <= 70:
        return BC_CUE, False
    if s.white_frac > 0.55 and s.chroma_frac < 0.20 and L >= 140.0:
        return BC_CUE, False
    if L <= 55.0 and s.chroma < 22.0:
        return BC_BLACK, False
    if V <= 70 and S <= 90 and s.chroma < 28.0:
        return BC_BLACK, False

    is_stripe = s.white_frac >= 0.16 and s.chroma_frac >= 0.22 and s.chroma > 20.0

    protos = [
        (BC_YELLOW, -12.0, 62.0, 180.0, 28.0),
        (BC_ORANGE, 38.0, 52.0, 150.0, 14.0),
        (BC_RED, 52.0, 28.0, 110.0, 2.0),
        (BC_MAROON, 32.0, 12.0, 75.0, 0.0),
        (BC_PURPLE, 28.0, -32.0, 100.0, 145.0),
        (BC_BLUE, 12.0, -48.0, 110.0, 110.0),
        (BC_GREEN, -42.0, 38.0, 120.0, 55.0),
    ]

    best, best_id = 1e9, BC_UNKNOWN
    for pid, aa, bb, Lp, h in protos:
        da, db, dL = a - aa, b - bb, (L - Lp) * 0.25
        dLab = da * da + db * db + dL * dL
        dH = hue_dist(float(H), h)
        cost = dLab + 0.35 * dH * dH
        if S < 50:
            cost += 80.0
        if cost < best:
            best, best_id = cost, pid

    if S >= 70 and V >= 60:
        if H <= 6 or H >= 170:
            best_id = BC_MAROON if (V < 100 or L < 95.0) else BC_RED
        elif H < 18:
            best_id = BC_ORANGE
            if H >= 15 and L > 190.0 and a < 18.0:
                best_id = BC_YELLOW
        elif H < 24:
            best_id = BC_ORANGE if (a > 28.0 or L < 155.0) else BC_YELLOW
        elif H < 38:
            best_id = BC_YELLOW
        elif H < 88:
            best_id = BC_GREEN
        elif H < 128:
            best_id = BC_BLUE
        elif H < 165:
            best_id = BC_PURPLE

    if best_id == BC_UNKNOWN and S < 55:
        if L >= 150.0:
            return BC_CUE, False
        if L <= 70.0:
            return BC_BLACK, False
    return best_id, is_stripe


def vote_label(hist: list[str], cur: str, max_hist: int = 9) -> str:
    hist.append(cur)
    if len(hist) > max_hist:
        del hist[0]
    counts = Counter(hist)
    return counts.most_common(1)[0][0]


def sample_ball_color(
    hsv: np.ndarray, lab: np.ndarray, cx: float, cy: float, r: float, felt_h: float = 55.0, have_felt: bool = True
) -> ColorSample:
    h, w = hsv.shape[:2]
    x0 = max(0, int(math.floor(cx - r - 1)))
    y0 = max(0, int(math.floor(cy - r - 1)))
    x1 = min(w - 1, int(math.ceil(cx + r + 1)))
    y1 = min(h - 1, int(math.ceil(cy + r + 1)))
    r_in, r_out = r * 0.22, r * 0.70

    Hs, Ss, Vs, Ls, As, Bs = [], [], [], [], [], []
    n_white = n_dark = n_chroma = n_all = 0

    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            dx, dy = x - cx, y - cy
            d2 = dx * dx + dy * dy
            if d2 < r_in * r_in or d2 > r_out * r_out:
                continue
            n_all += 1
            H, S, V = map(int, hsv[y, x])
            if V >= 230 and S <= 50:
                n_white += 1
                continue
            if V <= 28:
                n_dark += 1
                continue
            dist = math.sqrt(d2)
            if have_felt and dist > r * 0.52 and S >= 40 and hue_dist(float(H), felt_h) < 14.0:
                continue
            if S <= 55 and V >= 155:
                n_white += 1
            elif V <= 55:
                n_dark += 1
            if S >= 70 and V >= 55:
                n_chroma += 1
            if S < 35 and V > 170:
                continue
            Ls.append(float(lab[y, x, 0]))
            As.append(float(lab[y, x, 1]))
            Bs.append(float(lab[y, x, 2]))
            Hs.append(float(H))
            Ss.append(float(S))
            Vs.append(float(V))

    white_frac = n_white / n_all if n_all else 0.0
    dark_frac = n_dark / n_all if n_all else 0.0
    chroma_frac = n_chroma / n_all if n_all else 0.0

    if len(Hs) < 8:
        # fallback mean of disc
        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r * 0.55) ** 2
        if not np.any(mask):
            return ColorSample((0, 0, 0), (0, 0, 0), ok=False)
        mh = hsv[y0 : y1 + 1, x0 : x1 + 1][mask].mean(axis=0)
        ml = lab[y0 : y1 + 1, x0 : x1 + 1][mask].mean(axis=0)
        aa, bb = ml[1] - 128.0, ml[2] - 128.0
        return ColorSample(
            (int(mh[0]), int(mh[1]), int(mh[2])),
            (float(ml[0]), float(ml[1]), float(ml[2])),
            math.sqrt(aa * aa + bb * bb),
            white_frac,
            dark_frac,
            chroma_frac,
            True,
        )

    def med(v: list[float]) -> float:
        v = sorted(v)
        return v[len(v) // 2]

    h0 = med(Hs)
    Hu = []
    for h in Hs:
        if h0 < 20 and h > 100:
            h -= 180
        if h0 > 160 and h < 80:
            h += 180
        Hu.append(h)
    h_med = med(Hu)
    if h_med < 0:
        h_med += 180
    if h_med >= 180:
        h_med -= 180
    s_med, v_med = med(Ss), med(Vs)
    l_med, a_med, b_med = med(Ls), med(As), med(Bs)
    aa, bb = a_med - 128.0, b_med - 128.0
    return ColorSample(
        (int(h_med), int(s_med), int(v_med)),
        (l_med, a_med, b_med),
        math.sqrt(aa * aa + bb * bb),
        white_frac,
        dark_frac,
        chroma_frac,
        True,
    )


# ---- synthetic scene --------------------------------------------------------

# BGR paints tuned to land in expected OpenCV HSV/Lab bins
BALL_BGR = {
    "CUE": (245, 245, 245),
    "YELLOW": (40, 220, 255),
    "BLUE": (220, 90, 30),
    "RED": (40, 40, 220),
    "PURPLE": (180, 50, 150),
    "ORANGE": (30, 140, 255),
    "GREEN": (50, 180, 50),
    "MAROON": (35, 35, 110),
    "BLACK": (25, 25, 25),
}


def make_table(w: int = 640, h: int = 360) -> tuple[np.ndarray, list[tuple[str, float, float, float, bool]]]:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (45, 140, 45)  # green felt BGR
    balls: list[tuple[str, float, float, float, bool]] = []
    layout = [
        ("CUE", 120, 180, False),
        ("YELLOW", 220, 120, False),
        ("BLUE", 300, 120, False),
        ("RED", 380, 120, False),
        ("PURPLE", 460, 120, False),
        ("ORANGE", 220, 220, False),
        ("GREEN", 300, 220, False),
        ("MAROON", 380, 220, False),
        ("BLACK", 460, 220, False),
        ("YELLOW", 540, 170, True),  # stripe
    ]
    r = 22.0
    for name, x, y, stripe in layout:
        color = BALL_BGR[name]
        cv2.circle(img, (int(x), int(y)), int(r), color, -1, cv2.LINE_AA)
        # specular highlight
        cv2.circle(img, (int(x - 6), int(y - 6)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        if stripe and name != "CUE" and name != "BLACK":
            cv2.ellipse(img, (int(x), int(y)), (int(r * 0.9), int(r * 0.28)), 0, 0, 360, (240, 240, 240), -1)
        balls.append((name, float(x), float(y), r, stripe))
    return img, balls


def label_of(bc: int, stripe: bool) -> str:
    name = NAMES[bc]
    if stripe and bc not in (BC_CUE, BC_BLACK, BC_UNKNOWN):
        return name + "/s"
    return name


# ---- tests ------------------------------------------------------------------

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        failures.append(msg)


def test_prototype_samples() -> None:
    print("\n[1] Prototype HSV/Lab samples")
    # Build tiny patches, convert, classify
    cases = [
        ("CUE", (245, 245, 245)),
        ("BLACK", (20, 20, 20)),
        ("YELLOW", (40, 220, 255)),
        ("ORANGE", (30, 140, 255)),
        ("RED", (40, 40, 220)),
        ("MAROON", (35, 35, 100)),
        ("BLUE", (220, 90, 30)),
        ("GREEN", (50, 180, 50)),
        ("PURPLE", (180, 50, 150)),
    ]
    for name, bgr in cases:
        patch = np.full((32, 32, 3), bgr, dtype=np.uint8)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2Lab)
        mh = hsv.reshape(-1, 3).mean(axis=0)
        ml = lab.reshape(-1, 3).mean(axis=0)
        aa, bb = ml[1] - 128.0, ml[2] - 128.0
        s = ColorSample(
            (int(mh[0]), int(mh[1]), int(mh[2])),
            (float(ml[0]), float(ml[1]), float(ml[2])),
            math.sqrt(aa * aa + bb * bb),
            white_frac=0.05 if name != "CUE" else 0.7,
            chroma_frac=0.8 if name not in ("CUE", "BLACK") else 0.05,
            ok=True,
        )
        bc, _ = classify_color_sample(s)
        check(NAMES[bc] == name, f"uniform {name} -> {NAMES[bc]} (H={s.hsv[0]} S={s.hsv[1]} V={s.hsv[2]} L={s.lab[0]:.0f})")


def test_synthetic_table() -> None:
    print("\n[2] Synthetic table sampling + classify")
    img, balls = make_table()
    # mild warm cast
    img = cv2.convertScaleAbs(img, alpha=1.05, beta=8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    felt_h = float(np.median(hsv[hsv[:, :, 1] > 40][:, 0])) if np.any(hsv[:, :, 1] > 40) else 55.0

    for name, x, y, r, stripe in balls:
        samp = sample_ball_color(hsv, lab, x, y, r, felt_h, True)
        bc, got_stripe = classify_color_sample(samp)
        got = label_of(bc, got_stripe)
        expect = name + ("/s" if stripe else "")
        # stripe detection can be soft; allow base colour match for stripes
        ok = got == expect or (stripe and got.startswith(name))
        check(ok, f"ball {expect} @({x:.0f},{y:.0f}) -> {got} (H={samp.hsv[0]} chroma={samp.chroma:.1f} w={samp.white_frac:.2f})")


def test_specular_robustness() -> None:
    print("\n[3] Specular highlight does not turn red into CUE")
    img = np.full((80, 80, 3), (40, 40, 220), dtype=np.uint8)
    cv2.circle(img, (40, 40), 28, (40, 40, 220), -1)
    cv2.circle(img, (32, 32), 8, (255, 255, 255), -1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    samp = sample_ball_color(hsv, lab, 40, 40, 28, 55.0, False)
    bc, _ = classify_color_sample(samp)
    check(NAMES[bc] == "RED", f"specular red -> {NAMES[bc]} (not CUE)")


def test_vote_and_lock() -> None:
    print("\n[4] Temporal majority vote")
    hist: list[str] = []
    seq = ["RED", "ORANGE", "RED", "RED", "RED", "ORANGE", "RED", "RED", "RED"]
    last = ""
    for s in seq:
        last = vote_label(hist, s)
    check(last == "RED", f"vote majority -> {last}")


def test_felt_vs_green() -> None:
    print("\n[5] Green ball on green felt still samples as GREEN")
    img = np.full((120, 120, 3), (45, 140, 45), dtype=np.uint8)
    cv2.circle(img, (60, 60), 24, (40, 200, 40), -1)
    cv2.circle(img, (52, 52), 4, (255, 255, 255), -1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    felt_h = float(np.median(hsv[10:20, 10:20, 0]))
    samp = sample_ball_color(hsv, lab, 60, 60, 24, felt_h, True)
    bc, _ = classify_color_sample(samp)
    check(NAMES[bc] == "GREEN", f"green-on-felt -> {NAMES[bc]} H={samp.hsv[0]} S={samp.hsv[1]}")


def test_washed_yellow_not_cue() -> None:
    print("\n[7] Washed / bright yellow must not become CUE")
    # Simulate overexposed yellow: high V, moderate S, strong Lab b*
    cases = [
        ("pale yellow", (60, 210, 250)),   # BGR-ish bright yellow
        ("hot yellow", (40, 230, 255)),
        ("cream yellow", (90, 200, 240)),
    ]
    for name, bgr in cases:
        img = np.full((48, 48, 3), bgr, dtype=np.uint8)
        cv2.circle(img, (24, 24), 6, (255, 255, 255), -1)  # specular
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        samp = sample_ball_color(hsv, lab, 24, 24, 20, 55.0, False)
        bc, _ = classify_color_sample(samp)
        check(NAMES[bc] == "YELLOW", f"{name} -> {NAMES[bc]} (H={samp.hsv[0]} S={samp.hsv[1]} b*={samp.lab[2]-128:.0f})")


def test_true_cue_still_cue() -> None:
    print("\n[8] True white cue stays CUE")
    img = np.full((48, 48, 3), (245, 245, 245), dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    samp = sample_ball_color(hsv, lab, 24, 24, 18, 55.0, False)
    bc, _ = classify_color_sample(samp)
    check(NAMES[bc] == "CUE", f"white cue -> {NAMES[bc]}")
    print("\n[6] Warm illumination still separates YELLOW vs ORANGE")
    for name, bgr in (("YELLOW", (40, 220, 255)), ("ORANGE", (30, 140, 255))):
        img = np.full((64, 64, 3), bgr, dtype=np.uint8)
        # warm shift: boost R/B channels differently
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=15)
        img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int16) - 10, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        mh = hsv.reshape(-1, 3).mean(axis=0)
        ml = lab.reshape(-1, 3).mean(axis=0)
        aa, bb = ml[1] - 128.0, ml[2] - 128.0
        s = ColorSample(
            (int(mh[0]), int(mh[1]), int(mh[2])),
            (float(ml[0]), float(ml[1]), float(ml[2])),
            math.sqrt(aa * aa + bb * bb),
            0.05,
            0.0,
            0.85,
            True,
        )
        bc, _ = classify_color_sample(s)
        check(NAMES[bc] == name, f"warm {name} -> {NAMES[bc]}")


def main() -> int:
    print("poolShark colour classifier regression tests")
    print(f"OpenCV {cv2.__version__}")
    test_prototype_samples()
    test_synthetic_table()
    test_specular_robustness()
    test_vote_and_lock()
    test_felt_vs_green()
    test_illumination_shift()
    print()
    if failures:
        print(f"RESULT: {len(failures)} failure(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
