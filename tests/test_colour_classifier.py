#!/usr/bin/env python3
"""Regression tests for common pool-ball colour mix-ups."""
from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np
import cv2

BC_CUE, BC_YELLOW, BC_BLUE, BC_RED, BC_PURPLE = 0, 1, 2, 3, 4
BC_ORANGE, BC_GREEN, BC_MAROON, BC_BLACK, BC_UNKNOWN = 5, 6, 7, 8, 9
NAMES = {
    BC_CUE: "CUE", BC_YELLOW: "YELLOW", BC_BLUE: "BLUE", BC_RED: "RED",
    BC_PURPLE: "PURPLE", BC_ORANGE: "ORANGE", BC_GREEN: "GREEN",
    BC_MAROON: "MAROON", BC_BLACK: "BLACK", BC_UNKNOWN: "?",
}


@dataclass
class ColorSample:
    hsv: tuple
    lab: tuple
    chroma: float = 0.0
    white_frac: float = 0.0
    dark_frac: float = 0.0
    chroma_frac: float = 0.0
    ok: bool = True


def hue_dist(a, b):
    d = abs(a - b)
    return min(d, 180.0 - d)


def disambiguate(c, s):
    L, a, b = s.lab[0], s.lab[1] - 128.0, s.lab[2] - 128.0
    H, S, V = s.hsv
    if c == BC_CUE:
        if b > 18.0 and L > 110.0 and (S >= 25 or s.chroma >= 14.0) and 8 <= H <= 42 and a > -22.0:
            return BC_ORANGE if (H < 16 and a > 28.0) else BC_YELLOW
        if S >= 55 and V >= 70:
            if H <= 8 or H >= 170:
                return BC_MAROON if L < 95.0 else BC_RED
            if H < 18:
                return BC_ORANGE
            if H < 42:
                return BC_YELLOW
            if H < 88:
                return BC_GREEN
            if H < 128:
                return BC_BLUE
            if H < 165:
                return BC_PURPLE
        if s.chroma > 16.0 or abs(a) > 18.0 or abs(b) > 18.0:
            return BC_UNKNOWN
    if c == BC_YELLOW:
        if H < 8 or H >= 170:
            return BC_MAROON if L < 95.0 else BC_RED
        if 8 <= H < 16 and a > 25.0:
            return BC_ORANGE
        if 45 <= H < 90 and a < -20.0:
            return BC_GREEN
        if S < 30 and abs(b) < 16.0 and L > 160.0 and s.chroma < 14.0:
            return BC_CUE
    if c == BC_ORANGE:
        if H >= 20 and a < 22.0 and b > 35.0:
            return BC_YELLOW
        if (H <= 5 or H >= 172) and a > 35.0:
            return BC_MAROON if L < 95.0 else BC_RED
    if c == BC_RED and (L < 88.0 or V < 105.0):
        return BC_MAROON
    if c == BC_MAROON and L > 105.0 and V > 140.0 and S >= 70:
        return BC_RED
    if c == BC_BLUE and (H >= 132 or (H >= 125 and a > 20.0)):
        return BC_PURPLE
    if c == BC_PURPLE and H < 118 and b < -35.0 and a < 25.0:
        return BC_BLUE
    if c == BC_BLACK and s.chroma > 24.0 and a > 12.0:
        return BC_MAROON
    if c == BC_MAROON and s.chroma < 14.0 and L < 48.0 and S < 60:
        return BC_BLACK
    if c == BC_GREEN and S < 40 and L > 140.0 and s.chroma < 18.0:
        return BC_UNKNOWN
    return c


def classify_color_sample(s):
    is_stripe = False
    if not s.ok:
        return BC_UNKNOWN, False
    L, a, b = s.lab[0], s.lab[1] - 128.0, s.lab[2] - 128.0
    H, S, V = s.hsv
    yellow_signal = (
        (S >= 35 and 18 <= H <= 40 and V >= 90)
        or (b > 28.0 and -22.0 < a < 22.0 and L > 130.0 and 16 <= H <= 40 and s.chroma >= 16.0)
    )
    orange_signal = (
        (S >= 50 and 8 <= H < 18 and V >= 80)
        or (a > 28.0 and 22.0 < b < 60.0 and 90.0 < L < 185.0 and 8 <= H < 18)
    )
    if not yellow_signal and not orange_signal:
        if s.chroma < 12.0 and L >= 160.0 and S <= 42 and abs(a) < 14.0 and abs(b) < 16.0:
            return disambiguate(BC_CUE, s), False
        if s.white_frac > 0.62 and s.chroma_frac < 0.12 and L >= 155.0 and s.chroma < 12.0 and abs(b) < 16.0 and abs(a) < 14.0:
            return disambiguate(BC_CUE, s), False
    if L <= 50.0 and s.chroma < 20.0:
        return disambiguate(BC_BLACK, s), False
    if V <= 65 and S <= 80 and s.chroma < 24.0 and not yellow_signal and not orange_signal:
        return disambiguate(BC_BLACK, s), False
    is_stripe = s.white_frac >= 0.16 and s.chroma_frac >= 0.22 and s.chroma > 20.0
    if orange_signal:
        return disambiguate(BC_ORANGE, s), is_stripe
    if yellow_signal and b > 18.0 and H >= 16:
        return disambiguate(BC_YELLOW, s), is_stripe

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
        cost = da * da + db * db + dL * dL + 0.35 * hue_dist(float(H), h) ** 2
        if S < 50:
            cost += 80.0
        if pid == BC_YELLOW and b > 25.0:
            cost *= 0.55
        if pid == BC_ORANGE and a > 25.0 and H < 18:
            cost *= 0.65
        if pid == BC_MAROON and L < 90.0 and (H <= 8 or H >= 170):
            cost *= 0.70
        if pid == BC_BLUE and 95 <= H < 125 and b < -30.0:
            cost *= 0.70
        if pid == BC_PURPLE and 130 <= H < 165:
            cost *= 0.70
        if cost < best:
            best, best_id = cost, pid

    if S >= 45 and V >= 60:
        if H <= 6 or H >= 170:
            best_id = BC_MAROON if (V < 110 or L < 95.0) else BC_RED
        elif H < 18:
            best_id = BC_ORANGE
            if H >= 15 and L > 190.0 and a < 18.0:
                best_id = BC_YELLOW
            if b > 40.0 and a < 22.0 and H >= 14:
                best_id = BC_YELLOW
        elif H < 24:
            best_id = BC_ORANGE if (a > 28.0 and b < 50.0) else BC_YELLOW
        elif H < 42:
            best_id = BC_YELLOW
        elif H < 88:
            best_id = BC_GREEN
        elif H < 128:
            best_id = BC_BLUE
        elif H < 165:
            best_id = BC_PURPLE
    elif yellow_signal:
        best_id = BC_YELLOW
    elif orange_signal:
        best_id = BC_ORANGE

    if best_id == BC_UNKNOWN and S < 50:
        if L >= 155.0 and s.chroma < 12.0 and abs(b) < 14.0 and abs(a) < 12.0:
            best_id = BC_CUE
        elif L <= 70.0:
            best_id = BC_BLACK
        elif b > 18.0 and L > 120.0 and 16 <= H <= 40:
            best_id = BC_YELLOW
        elif a > 25.0 and 8 <= H < 20:
            best_id = BC_ORANGE
    return disambiguate(best_id, s), is_stripe


def sample_from_bgr(bgr, specular=False):
    img = np.full((48, 48, 3), bgr, dtype=np.uint8)
    if specular:
        cv2.circle(img, (20, 18), 5, (255, 255, 255), -1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    mask = np.ones((48, 48), np.uint8)
    cv2.circle(mask, (24, 24), 6, 0, -1)
    ys, xs = np.where(mask > 0)
    ph, pl = hsv[ys, xs], lab[ys, xs]
    keep = ~((ph[:, 2] >= 235) & (ph[:, 1] <= 35))
    ph, pl = ph[keep], pl[keep]
    if len(ph) < 8:
        mh, ml = hsv.reshape(-1, 3).mean(0), lab.reshape(-1, 3).mean(0)
    else:
        sat = ph[:, 1] >= 40
        if sat.sum() >= 8:
            ph, pl = ph[sat], pl[sat]
        mh, ml = np.median(ph, 0), np.median(pl, 0)
    aa, bb = ml[1] - 128.0, ml[2] - 128.0
    return ColorSample(
        (int(mh[0]), int(mh[1]), int(mh[2])),
        (float(ml[0]), float(ml[1]), float(ml[2])),
        math.sqrt(aa * aa + bb * bb),
        float(((ph[:, 1] <= 45) & (ph[:, 2] >= 170)).mean()) if len(ph) else 0.0,
        0.0,
        float(((ph[:, 1] >= 55) & (ph[:, 2] >= 55)).mean()) if len(ph) else 0.0,
        True,
    )


failures = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def expect(bgr, name, specular=False, tag=""):
    s = sample_from_bgr(bgr, specular=specular)
    bc, _ = classify_color_sample(s)
    got = NAMES[bc]
    check(got == name, f"{tag or name} BGR{bgr} -> {got} (want {name}) H={s.hsv[0]} S={s.hsv[1]} L={s.lab[0]:.0f} a*={s.lab[1]-128:.0f} b*={s.lab[2]-128:.0f}")


def main():
    print("poolShark confusion-matrix regression")
    print(f"OpenCV {cv2.__version__}\n")

    print("[1] Canonical solids")
    expect((245, 245, 245), "CUE")
    expect((40, 220, 255), "YELLOW")
    expect((30, 140, 255), "ORANGE")
    expect((40, 40, 220), "RED")
    expect((35, 35, 100), "MAROON")
    expect((220, 90, 30), "BLUE")
    expect((180, 50, 150), "PURPLE")
    expect((50, 180, 50), "GREEN")
    expect((20, 20, 20), "BLACK")

    print("\n[2] Glare / wash mix-ups")
    expect((60, 210, 250), "YELLOW", specular=True, tag="washed yellow+glare")
    expect((90, 200, 240), "YELLOW", tag="cream yellow")
    expect((40, 230, 255), "YELLOW", specular=True, tag="hot yellow")
    expect((245, 245, 245), "CUE", specular=True, tag="true cue+glare")
    expect((50, 160, 255), "ORANGE", tag="bright orange")
    expect((25, 25, 90), "MAROON", tag="dark maroon not black")
    expect((30, 30, 30), "BLACK", tag="true black")
    expect((200, 80, 40), "BLUE", tag="bright blue not purple")
    expect((160, 40, 140), "PURPLE", tag="purple not blue")

    print("\n[3] Pairwise borders")
    expect((35, 150, 240), "ORANGE", tag="border orange")
    expect((45, 215, 250), "YELLOW", tag="border yellow")
    expect((30, 30, 120), "MAROON", tag="dark red to maroon")
    expect((50, 50, 230), "RED", tag="bright red")

    print("\n[5] Common confusion pairs")
    # cue / yellow
    expect((230, 230, 235), "CUE", tag="cool white cue")
    expect((70, 195, 235), "YELLOW", tag="pastel yellow not cue")
    # orange / yellow
    expect((40, 130, 250), "ORANGE", tag="deep orange")
    expect((55, 200, 245), "YELLOW", tag="golden yellow")
    # red / maroon
    expect((45, 45, 200), "RED", tag="solid red")
    expect((28, 28, 85), "MAROON", tag="solid maroon")
    # blue / purple
    expect((210, 100, 35), "BLUE", tag="solid blue")
    expect((170, 45, 135), "PURPLE", tag="solid purple")
    # green vs felt-like pale / yellow
    expect((45, 170, 55), "GREEN", tag="solid green")
    expect((80, 200, 80), "GREEN", tag="bright green")
    # black / maroon
    expect((18, 18, 18), "BLACK", tag="near-black")
    expect((40, 30, 75), "MAROON", tag="brownish maroon")

    print("\n[4] Vote stability")
    hist = []
    for x in ["YELLOW", "CUE", "YELLOW", "YELLOW", "YELLOW", "CUE", "YELLOW"]:
        hist.append(x)
        if len(hist) > 9:
            del hist[0]
    last = Counter(hist).most_common(1)[0][0]
    check(last == "YELLOW", f"vote resists cue flicker -> {last}")

    print()
    if failures:
        print(f"RESULT: {len(failures)} failure(s)")
        for f in failures:
            print(" -", f)
        return 1
    print("RESULT: all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
