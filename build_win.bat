@echo off
setlocal
call "E:\BuildTools\VS2022\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1

set QTDIR=E:\deps\Qt\5.12.12\msvc2017_64
set PATH=%QTDIR%\bin;E:\deps\opencv\opencv\build\x64\vc16\bin;%PATH%
set OPENCV_DIR=E:\deps\opencv\opencv\build

cd /d "E:\Freelancer project\2026\10\poolShark"
if not exist build mkdir build
cd build

echo === qmake ===
"%QTDIR%\bin\qmake.exe" ..\poolShark.pro -spec win32-msvc "CONFIG+=release" || exit /b 1

echo === nmake ===
nmake || exit /b 1

echo === build ok ===
dir /b release\*.exe 2>nul
dir /b *.exe 2>nul
endlocal
