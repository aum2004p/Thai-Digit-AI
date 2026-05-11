#!/bin/bash
echo "============================================"
echo " Thai Digit Recognition - Setup and Run"
echo "============================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python 3.9+"
    exit 1
fi

echo "[1/4] Installing dependencies..."
pip3 install -r requirements.txt || { echo "[ERROR] Failed to install dependencies"; exit 1; }

echo
echo "[2/4] Generating dataset..."
python3 dataset/generate_dataset.py || echo "[WARNING] Dataset generation had issues."

echo
echo "[3/4] Training model..."
python3 training/train_model.py || echo "[WARNING] Training failed. Web app will use mock predictions."

echo
echo "[4/4] Starting web application..."
echo
echo " Open browser at: http://localhost:5000"
echo " Admin page:      http://localhost:5000/admin"
echo " Admin login:     admin / admin1234"
echo
cd webapp && python3 app.py
