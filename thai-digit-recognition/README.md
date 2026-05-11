# 🔢 Thai Digit Recognition (51–55)

ระบบจดจำเลขไทยลายมือ 51–55 ด้วย CNN + Flask Web Application

---

## 📁 โครงสร้างโปรเจกต์

```
thai-digit-recognition/
├── dataset/
│   ├── generate_dataset.py   # สร้าง dataset เลขไทย 51-55
│   ├── train/                # รูปสำหรับ train (80 รูป/เลข)
│   └── test/                 # รูปสำหรับ test  (20 รูป/เลข)
├── training/
│   └── train_model.py        # เทรน CNN model
├── model/
│   ├── thai_digit_model.keras # trained model
│   ├── best_model.keras       # best checkpoint
│   └── model_metadata.json   # accuracy, labels, etc.
├── webapp/
│   ├── app.py                # Flask application
│   ├── templates/
│   │   ├── base.html
│   │   ├── user.html         # หน้าเขียนเลข + ทำนาย
│   │   ├── admin_login.html  # หน้า login admin
│   │   └── admin.html        # หน้า admin dashboard
│   └── uploads/              # โมเดลที่ admin อัปโหลด
├── .vscode/
│   ├── launch.json           # Debug configurations
│   ├── tasks.json            # Build tasks
│   └── settings.json
├── requirements.txt
├── setup_and_run.bat         # Windows one-click setup
└── setup_and_run.sh          # Linux/Mac one-click setup
```

---

## 🚀 วิธีติดตั้งและรัน

### วิธีที่ 1: One-click (Windows)
```
double-click setup_and_run.bat
```

### วิธีที่ 2: Manual

**1. ติดตั้ง dependencies**
```bash
pip install -r requirements.txt
```

**2. สร้าง dataset** (ต้องมี Thai font บนเครื่อง)
```bash
python dataset/generate_dataset.py
```

**3. เทรนโมเดล**
```bash
python training/train_model.py
```

**4. รัน web app**
```bash
cd webapp
python app.py
```

เปิด browser ที่ http://localhost:5000

---

## 🖥️ หน้าเว็บ

| URL | คำอธิบาย |
|-----|----------|
| `http://localhost:5000/` | หน้าผู้ใช้ - เขียนเลข + ทำนาย |
| `http://localhost:5000/admin` | Admin dashboard |
| `http://localhost:5000/admin/login` | Admin login |

**Admin credentials:** `admin` / `admin1234`

---

## 🧠 Model Architecture (PyTorch CNN)

```
Input: 32×32 grayscale image (1 channel)

Conv2d(1→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)   [16×16]
Conv2d(32→64,3×3) → BatchNorm → ReLU → MaxPool(2×2)   [8×8]
Conv2d(64→128,3×3)→ BatchNorm → ReLU → AdaptiveAvgPool [128-dim]
Linear(128→128) → ReLU → Dropout(0.4)
Linear(128→5)   → Softmax

Total params: ~110K
Framework: PyTorch 2.x (supports Python 3.14)
```

---

## 📊 Dataset

- **เลข:** ๕๑ ๕๒ ๕๓ ๕๔ ๕๕ (51–55)
- **จำนวน:** 100 รูป/เลข = 500 รูปรวม
- **แบ่ง:** 80 train / 20 test ต่อเลข
- **ขนาด:** 32×32 pixels, grayscale
- **ความหลากหลาย:**
  - ตำแหน่ง: กลาง, บนซ้าย, บนขวา, ล่างซ้าย, ล่างขวา, ฯลฯ
  - การหมุน: -20° ถึง +20°
  - ขนาดตัวอักษร: หลายขนาด
  - Noise และ Blur
  - ความสว่างและ contrast ต่างกัน

---

## ⚙️ VS Code Tasks

เปิด Command Palette (`Ctrl+Shift+P`) → `Tasks: Run Task`

- **Install Dependencies** - ติดตั้ง pip packages
- **Generate Dataset** - สร้างรูปภาพ dataset
- **Train Model** - เทรน CNN
- **Run Web App** - รัน Flask server
- **Full Setup** - ทำทุกขั้นตอนต่อเนื่อง

---

## 📝 หมายเหตุ

- ใช้ **PyTorch** (รองรับ Python 3.14+) แทน TensorFlow
- ต้องมี **Thai font** บนเครื่อง (เช่น Leelawad, Tahoma) สำหรับสร้าง dataset — ถ้าไม่มีจะใช้ stroke-based rendering แทน
- เปลี่ยน `ADMIN_PASSWORD_HASH` ใน `app.py` ก่อน deploy จริง
- โมเดลไฟล์: `.pt` (PyTorch state dict) หรือ `.pth`
