@echo off
title Stage 2 - Data Augmentation
color 0B

echo.
echo ============================================
echo   STAGE 2 — Data Augmentation
echo   Blur Filter -^> Generate Training Variants
echo ============================================
echo.
echo  Reads from:  Data_Augmentation\Data_Gathering\^<person_name^>\
echo  Outputs to:  Data_Augmentation\Augmented_Data\^<person_name^>\
echo.
echo  Step 2a — Remove blurry images (Laplacian threshold 50)
echo  Step 2b — Generate 40-300x augmented variants per image
echo.
pause

:: Activate conda environment
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
    echo         Run: pip install opencv-python albumentations numpy
    echo.
    pause
    exit /b 1
)

cd /d "%~dp0Data_Augmentation"
echo [OK] Activated cctv_face_recog
echo.

:: Step 2a - Blur filter
echo ----------------------------------------
echo  Step 2a: Removing blurry images...
echo ----------------------------------------
python leplace_detection.py
IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Blur filter failed. Check Data_Gathering has images.
    pause
    exit /b 1
)

echo.
echo ----------------------------------------
echo  Step 2b: Generating augmented dataset...
echo ----------------------------------------
python augment_faces.py
IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Augmentation failed.
    pause
    exit /b 1
)

echo.
echo [DONE] Augmentation complete.
echo        Output is in: Data_Augmentation\Augmented_Data\
echo.
echo  Next step: run stage3_recognize.bat to train the model
echo             and start live recognition.
echo.
pause
