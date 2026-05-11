@echo off
chcp 65001 >nul
echo ============================================
echo  Thai Digit Recognition - Setup and Run
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

echo [1/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/4] Generating dataset...
python dataset/generate_dataset.py
if errorlevel 1 (
    echo [WARNING] Dataset generation had issues. Check Thai font availability.
)

echo.
echo [3/4] Training model...
python training/train_model.py
if errorlevel 1 (
    echo [WARNING] Training failed. Web app will use mock predictions.
)

echo.
echo [4/4] Starting web application...
echo.
echo  Open browser at: http://localhost:5000
echo  Admin page:      http://localhost:5000/admin
echo  Admin login:     admin / admin1234
echo.
cd webapp
python app.py
pause
