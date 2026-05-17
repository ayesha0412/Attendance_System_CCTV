@echo off
title Stage 1 - Data Collection
color 0A

echo.
echo ============================================
echo   STAGE 1 — Data Collection
echo   RTSP Camera -^> Face Detection -^> Crops
echo ============================================
echo.
echo  This will connect to the CCTV camera and
echo  save face crops to:
echo  Data_Collection\data\raw_faces\
echo.
echo  Dashboard will open at: http://localhost:5000
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
    echo         Run: conda create -n cctv_face_recog python=3.10
    echo         Then: pip install -r Data_Collection\requirements.txt
    echo.
    pause
    exit /b 1
)

:: Move into Data_Collection and run
cd /d "%~dp0Data_Collection"
echo [OK] Activiated cctv_face_recog
echo [..] Starting data collection...
echo.

python main.py

echo.
echo [DONE] Collection stopped.
echo        Crops saved in: Data_Collection\data\raw_faces\
echo.
echo  Next step: sort your crops into named folders inside
echo             Data_Augmentation\Data_Gathering\^<person_name^>\
echo             Then run stage2_augment.bat
echo.
pause
