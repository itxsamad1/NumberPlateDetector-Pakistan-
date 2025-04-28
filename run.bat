@echo off
echo NumberPlateDetector (Pakistan)
echo ==================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.6 or higher from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Install required packages
echo Installing required packages...
pip install opencv-python PyQt5
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install required packages
    echo Please check the error message above.
    pause
    exit /b 1
)

echo.
echo Verifying required files...

REM Check if the video file exists
if not exist "outVideo.avi" (
    echo Warning: Video file 'outVideo.avi' not found.
    echo The application will still run, but you'll need to provide a video file.
    echo.
    timeout /t 5
)

REM Check if the cascade file exists
if not exist "pak.xml" (
    echo Error: Cascade file 'pak.xml' not found.
    echo This file is required for license plate detection.
    pause
    exit /b 1
)

echo.
echo Starting the application...
echo Use the "Load Video" or "Load Image" buttons to select files for processing

REM Run the application
python plate_detector.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application crashed or failed to start.
    echo Please check the error message above.
    pause
)

echo Application closed
pause 