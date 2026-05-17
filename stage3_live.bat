@echo off
title Stage 3b - Live Recognition
color 0D

echo.
echo ============================================
echo   STAGE 3b — Live Face Recognition
echo   RTSP Camera -^> Identify Employees
echo ============================================
echo.
echo  Dashboard will open at: http://localhost:5001
echo.

IF NOT EXIST "%~dp0Recognition\face_model.pkl" (
    echo [ERROR] No trained model found.
    echo         Run stage3_train.bat first.
    echo.
    pause
    exit /b 1
)

IF /I "%CONDA_DEFAULT_ENV%"=="cctv_face_recog" (
    echo [OK] Conda env already active: cctv_face_recog
) ELSE (
    set "CONDA_BAT=%UserProfile%\anaconda3\condabin\conda.bat"
    IF EXIST "%CONDA_BAT%" (
        CALL "%CONDA_BAT%" activate cctv_face_recog
    ) ELSE (
        CALL conda activate cctv_face_recog
    )
)

IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Could not activate conda environment: cctv_face_recog
    echo         Try: conda init cmd.exe
    pause
    exit /b 1
)

cd /d "%~dp0Recognition"
echo [OK] Activated cctv_face_recog
echo [OK] Model found: Recognition\face_model.pkl
echo.
echo  Starting... open http://localhost:5001 in your browser.
echo  Press Ctrl+C to stop.
echo.

python live_recognition.py

echo.
echo [DONE] Recognition stopped.
echo.
pause
