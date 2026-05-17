@echo off
title AdaFace Setup
color 0E

echo.
echo ============================================
echo   AdaFace Setup (run once)
echo   Downloads model architecture + weights
echo ============================================
echo.
echo  This will download:
echo    - net.py        (~20 KB, from GitHub)
echo    - adaface_ir50_ms1mv3.ckpt  (~167 MB, from Google Drive)
echo.
echo  Requires internet connection.
echo.
pause

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
python setup_adaface.py

IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Setup failed. See messages above.
    pause
    exit /b 1
)

echo.
echo [DONE] AdaFace ready.
echo        Run stage3_adaface_train.bat next.
echo.
pause
