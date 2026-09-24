# poolShark

AR pool-shot assistant for camera-equipped recording glasses.
Built with **Qt 5.12** and **OpenCV 4** (C++14).

## Colour classification (current)

Ball colours are no longer a single HSV average. The pipeline now:

1. **Illumination correction** — mild felt-based white balance + CLAHE on Lab `L`
2. **Robust sampling** — annular interior mask; drops specular glints, deep shadows, and felt bleed at the rim; uses medians (hue-wrap aware)
3. **Lab + HSV prototypes** — nearest Lab `a*b*` prototype with HSV tie-breaks for warm hues (yellow/orange/red/maroon)
4. **Stripe detection** — white-band fraction → labels like `YELLOW/s`
5. **Temporal lock** — majority vote over ~9 frames, then lock until sustained disagreement
6. **Extra detection** — Lab distance from felt + Hough circles so green balls are not lost in the cloth

Reset the tracker after large lighting changes.

## Build

```bash
# Point OPENCV_DIR at your OpenCV install, then:
qmake poolShark.pro
make          # or nmake / mingw32-make on Windows
```

On Windows with the official OpenCV package:

```bat
set OPENCV_DIR=C:\opencv\build
qmake poolShark.pro
nmake
```

Adjust the `opencv_world412` lib name in `poolShark.pro` if your OpenCV version differs.

## Tests

Synthetic colour-classifier regression (Python + OpenCV, no Qt required):

```bash
py -3 -m pip install opencv-python-headless numpy
py -3 tests/test_colour_classifier.py
```

## WIP

Still a demo: no camera calibration, 2D image-space tracking, simple ghost-ball physics.
