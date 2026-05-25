@echo off
title AdaFace - Train Gallery
color 0E

echo.
echo ============================================
echo   AdaFace - Build Recognition Gallery
echo   Reads from: Data_Augmentation\Dataset_Clicked\
echo   Saves to:   Recognition\face_model_adaface.pkl
echo ============================================
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
python train_adaface.py

IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Training failed. See messages above.
    pause
    exit /b 1
)

echo.
echo [DONE] Run stage3_adaface_live.bat to start recognition.
echo.
pause
