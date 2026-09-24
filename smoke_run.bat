@echo off
setlocal
set QTDIR=E:\deps\Qt\5.12.12\msvc2017_64
set PATH=%QTDIR%\bin;E:\deps\opencv\opencv\build\x64\vc16\bin;%PATH%
set QT_LOGGING_RULES=*.debug=false
cd /d "E:\Freelancer project\2026\10\poolShark\build\release"
echo Starting poolShark.exe ...
start "" /b poolShark.exe
timeout /t 4 /nobreak >nul
tasklist /FI "IMAGENAME eq poolShark.exe" | findstr /I poolShark.exe
if errorlevel 1 (
  echo FAIL: process not running
  exit /b 1
)
echo PASS: poolShark.exe is running
taskkill /IM poolShark.exe /F >nul 2>&1
echo Smoke test done
endlocal
