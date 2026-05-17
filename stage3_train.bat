@echo off
title Stage 3a - Train Model
color 0E

echo.
echo ============================================
echo   STAGE 3a — Train Recognition Model
echo   ArcFace Embeddings -^> SVM Classifier
echo ============================================
echo.
echo  Reads from:  Data_Augmentation\Augmented_Data\
echo  Saves to:    Recognition\face_model.pkl
echo.
echo  Run this whenever you:
echo    - Add a new employee
echo    - Re-run augmentation with more images
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
    echo.
    echo [ERROR] Could not activate conda environment: cctv_face_recog
    echo         Try: conda init cmd.exe
    pause
    exit /b 1
)

cd /d "%~dp0Recognition"
echo [OK] Activated cctv_face_recog
echo.

python train_embeddings.py
IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Training failed.
    echo         Make sure Data_Augmentation\Augmented_Data\ has images.
    echo         Run stage2_augment.bat first.
    pause
    exit /b 1
)

echo.
echo [DONE] Model saved to Recognition\face_model.pkl
echo        Run stage3_live.bat to start live recognition.
echo.
pause
