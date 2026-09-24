# poolShark

AR pool-shot assistant for camera-equipped recording glasses.
Built with **Qt 5.12** and **OpenCV 4** (C++11/14).

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
qmake poolShark.pro && make
```

Adjust `INCLUDEPATH` / `LIBS` in `poolShark.pro` for your OpenCV install.

## WIP

Still a demo: no camera calibration, 2D image-space tracking, simple ghost-ball physics.
