@echo off
title AdaFace - Live Recognition
color 0E

echo.
echo ============================================
echo   AdaFace - Live Face Recognition
echo   Dashboard: http://localhost:5002
echo ============================================
echo.

IF NOT EXIST "%~dp0Recognition\face_model_adaface.pkl" (
    echo [ERROR] No AdaFace model found.
    echo         Run stage3_adaface_train.bat first.
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
    echo [ERROR] Could not activate conda environment.
    pause
    exit /b 1
)

cd /d "%~dp0Recognition"
echo [OK] Model found: face_model_adaface.pkl
echo.
echo  Open http://localhost:5002 in your browser.
echo  Press Ctrl+C to stop.
echo.

python live_adaface.py

echo.
echo [DONE] Stopped.
echo.
pause
