QT       += widgets
TARGET    = poolShark
TEMPLATE  = app
CONFIG   += c++14

# Portable OpenCV paths:
#   Windows: set OPENCV_DIR to your OpenCV build (contains include/ and lib/ or x64/vc15/lib)
#   macOS/Linux: set OPENCV_DIR, or rely on pkg-config / system includes below.
isEmpty(OPENCV_DIR) {
    OPENCV_DIR = $$(OPENCV_DIR)
}

!isEmpty(OPENCV_DIR) {
    INCLUDEPATH += $$OPENCV_DIR/include
    INCLUDEPATH += $$OPENCV_DIR/include/opencv4
    win32 {
        # Official OpenCV Windows packages use x64/vc15|vc16|vc17/lib
        LIBS += -L$$OPENCV_DIR/x64/vc17/lib
        LIBS += -L$$OPENCV_DIR/x64/vc16/lib
        LIBS += -L$$OPENCV_DIR/x64/vc15/lib
        LIBS += -L$$OPENCV_DIR/lib
        CONFIG(debug, debug|release) {
            LIBS += -lopencv_world412d
        } else {
            LIBS += -lopencv_world412
        }
        # Fallback names if world lib version differs — uncomment/adjust:
        # LIBS += -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio
    } else {
        LIBS += -L$$OPENCV_DIR/lib -L$$OPENCV_DIR/build/lib
        LIBS += -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio
    }
} else:unix {
    # System packages (Debian/Ubuntu/Homebrew opencv4)
    INCLUDEPATH += /usr/include/opencv4 /usr/local/include/opencv4
    LIBS += -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio
}

# Optional: avoid linking highgui (can conflict with Qt on some builds)
# LIBS += -lopencv_highgui

SOURCES += main.cpp
