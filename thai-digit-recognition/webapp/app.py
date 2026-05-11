"""
Flask Web Application for Thai Digit Recognition (51-55)
- User page: draw digit, predict
- Admin page: upload new model
Uses PyTorch for inference.
"""

import os
import io
import sys
import json
import base64
import logging
from datetime import datetime
from functools import wraps

import numpy as np
from PIL import Image
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, flash)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# ── PyTorch import ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Prediction will return mock results.")

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'thai-digit-secret-key-change-in-prod')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, '..', 'model')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

ALLOWED_MODEL_EXT   = {'.pt', '.pth'}
ALLOWED_IMAGE_EXT   = {'.png', '.jpg', '.jpeg'}
IMG_SIZE = 32
LABELS   = [51, 52, 53, 54, 55]

SAVED_DRAWINGS_DIR = os.path.join(BASE_DIR, 'saved_drawings')
os.makedirs(SAVED_DRAWINGS_DIR, exist_ok=True)
THAI_DIGITS_MAP = {
    51: '๕๑', 52: '๕๒', 53: '๕๓', 54: '๕๔', 55: '๕๕'
}

# ── Admin credentials ──────────────────────────────────────────────────────────
ADMIN_USERNAME     = 'admin'
ADMIN_PASSWORD_HASH = generate_password_hash('admin1234')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Shared preprocessing (must match training pipeline) ───────────────────────
def preprocess_for_model(img_gray: Image.Image) -> Image.Image:
    """
    1. Invert if background is dark (normalize to dark-on-white)
    2. Threshold to binary
    3. Crop tight to the drawn content with padding
    4. Resize to IMG_SIZE x IMG_SIZE
    """
    arr = np.array(img_gray, dtype=np.uint8)

    # If mean pixel is dark → invert (light-on-dark → dark-on-light)
    if arr.mean() < 128:
        arr = 255 - arr

    # Binary threshold: anything darker than 200 is ink
    binary = (arr < 200).astype(np.uint8)

    # Find bounding box of ink pixels
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)

    if not rows.any():
        # Blank canvas — return white image
        return Image.fromarray(np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add padding (10% of bounding box size, min 4px)
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    pad = max(4, int(max(h, w) * 0.15))

    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(arr.shape[1] - 1, cmax + pad)

    cropped = arr[rmin:rmax+1, cmin:cmax+1]

    # Make square by padding shorter side
    ch, cw = cropped.shape
    side = max(ch, cw)
    square = np.ones((side, side), dtype=np.uint8) * 255
    top  = (side - ch) // 2
    left = (side - cw) // 2
    square[top:top+ch, left:left+cw] = cropped

    img_out = Image.fromarray(square).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return img_out


# ── CNN Architecture (must match training) ─────────────────────────────────────
if TORCH_AVAILABLE:
    class ThaiDigitCNN(nn.Module):
        def __init__(self, num_classes: int = 5):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x


# ── Model manager ──────────────────────────────────────────────────────────────
class ModelManager:
    def __init__(self):
        self.model = None
        self.metadata = {}
        self.model_path = None
        self.loaded_at = None
        self._load_default()

    def _load_default(self):
        candidates = [
            os.path.join(MODEL_DIR, 'thai_digit_model.pt'),
            os.path.join(MODEL_DIR, 'best_model.pt'),
            os.path.join(MODEL_DIR, 'thai_digit_scripted.pt'),
        ]
        for path in candidates:
            if os.path.exists(path):
                if self.load_model(path):
                    return
        logger.warning("No trained model found. Run training first.")

    def load_model(self, path: str) -> bool:
        if not TORCH_AVAILABLE:
            return False
        try:
            ext = os.path.splitext(path)[1].lower()
            # Try TorchScript first
            try:
                self.model = torch.jit.load(path, map_location=DEVICE)
                self.model.eval()
            except Exception:
                # Fall back to state dict
                checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    num_classes = checkpoint.get('num_classes', 5)
                    m = ThaiDigitCNN(num_classes=num_classes).to(DEVICE)
                    m.load_state_dict(checkpoint['model_state_dict'])
                    m.eval()
                    self.model = m
                    # Load embedded metadata
                    for key in ('labels', 'label_to_idx', 'idx_to_label',
                                'img_size', 'test_accuracy'):
                        if key in checkpoint:
                            self.metadata[key] = checkpoint[key]
                else:
                    raise ValueError("Unknown checkpoint format")

            self.model_path = path
            self.loaded_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Load external metadata if available
            meta_path = os.path.join(MODEL_DIR, 'model_metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.metadata.update(json.load(f))

            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            return False

    def predict(self, img_array: np.ndarray):
        """
        img_array: numpy array (H, W) grayscale 0-255
        Crops to the drawn content bounding box before resizing,
        matching the preprocessing used during training.
        Returns: (predicted_label, confidence, all_probs_list)
        """
        if self.model is None or not TORCH_AVAILABLE:
            import random
            probs = np.random.dirichlet(np.ones(5) * 0.5)
            idx = np.argmax(probs)
            return LABELS[idx], float(probs[idx]), probs.tolist()

        img = Image.fromarray(img_array.astype(np.uint8)).convert('L')
        img = preprocess_for_model(img)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        idx = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx]), probs.tolist()

    def get_info(self):
        return {
            'loaded':       self.model is not None,
            'path':         os.path.basename(self.model_path) if self.model_path else None,
            'loaded_at':    self.loaded_at,
            'accuracy':     self.metadata.get('test_accuracy'),
            'torch_available': TORCH_AVAILABLE,
            'torch_version':   torch.__version__ if TORCH_AVAILABLE else None,
        }


model_manager = ModelManager()


# ── Auth ───────────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('กรุณาเข้าสู่ระบบก่อน', 'warning')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated


# ── User routes ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('user.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'}), 400

        img_data = data['image']
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)

        img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img_gray = background.convert('L')
        img_array = np.array(img_gray)

        label, confidence, all_probs = model_manager.predict(img_array)

        # Auto-save drawing to saved_drawings/
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        save_name = f'drawing_{ts}.png'
        save_path = os.path.join(SAVED_DRAWINGS_DIR, save_name)
        img_gray.save(save_path)

        result = {
            'label':      label,
            'thai':       THAI_DIGITS_MAP[label],
            'confidence': round(confidence * 100, 2),
            'saved_as':   save_name,
            'all_probs':  [
                {
                    'label': LABELS[i],
                    'thai':  THAI_DIGITS_MAP[LABELS[i]],
                    'prob':  round(float(p) * 100, 2),
                }
                for i, p in enumerate(all_probs)
            ],
        }
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


# ── Admin routes ───────────────────────────────────────────────────────────────
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session['admin_logged_in'] = True
            session['admin_user'] = username
            flash('เข้าสู่ระบบสำเร็จ', 'success')
            return redirect(url_for('admin_dashboard'))
        flash('ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง', 'danger')
    return render_template('admin_login.html')


@app.route('/admin/logout')
def admin_logout():
    session.clear()
    flash('ออกจากระบบแล้ว', 'info')
    return redirect(url_for('admin_login'))


@app.route('/admin')
@login_required
def admin_dashboard():
    model_info = model_manager.get_info()
    uploaded_models = []
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if os.path.isfile(fpath):
            uploaded_models.append({
                'name':     fname,
                'size':     os.path.getsize(fpath),
                'modified': datetime.fromtimestamp(
                    os.path.getmtime(fpath)
                ).strftime('%Y-%m-%d %H:%M'),
            })
    uploaded_models.sort(key=lambda x: x['modified'], reverse=True)
    return render_template('admin.html',
                           model_info=model_info,
                           uploaded_models=uploaded_models)


@app.route('/admin/upload', methods=['POST'])
@login_required
def admin_upload_model():
    if 'model_file' not in request.files:
        flash('ไม่พบไฟล์', 'danger')
        return redirect(url_for('admin_dashboard'))

    file = request.files['model_file']
    if file.filename == '':
        flash('ไม่ได้เลือกไฟล์', 'danger')
        return redirect(url_for('admin_dashboard'))

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_MODEL_EXT:
        flash(f'ไฟล์ต้องเป็น .pt หรือ .pth เท่านั้น (ได้รับ: {ext})', 'danger')
        return redirect(url_for('admin_dashboard'))

    filename  = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    activate = request.form.get('activate') == 'on'
    if activate:
        success = model_manager.load_model(save_path)
        if success:
            flash(f'อัปโหลดและโหลดโมเดล "{filename}" สำเร็จ', 'success')
        else:
            flash(f'อัปโหลดสำเร็จ แต่โหลดโมเดลล้มเหลว (ตรวจสอบ log)', 'warning')
    else:
        flash(f'อัปโหลดไฟล์ "{filename}" สำเร็จ', 'success')

    return redirect(url_for('admin_dashboard'))


@app.route('/admin/activate/<filename>', methods=['POST'])
@login_required
def admin_activate_model(filename):
    safe_name  = secure_filename(filename)
    model_path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, safe_name)
    if not os.path.exists(model_path):
        flash('ไม่พบไฟล์โมเดล', 'danger')
        return redirect(url_for('admin_dashboard'))

    success = model_manager.load_model(model_path)
    if success:
        flash(f'โหลดโมเดล "{safe_name}" สำเร็จ', 'success')
    else:
        flash('โหลดโมเดลล้มเหลว', 'danger')
    return redirect(url_for('admin_dashboard'))


# ── Admin: predict from uploaded PNG ──────────────────────────────────────────
@app.route('/admin/predict-image', methods=['POST'])
@login_required
def admin_predict_image():
    if 'image_file' not in request.files:
        return jsonify({'error': 'ไม่พบไฟล์รูปภาพ'}), 400

    file = request.files['image_file']
    if file.filename == '':
        return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        return jsonify({'error': 'รองรับเฉพาะ .png, .jpg, .jpeg'}), 400

    try:
        img = Image.open(file.stream).convert('RGBA')
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img_gray  = background.convert('L')
        img_array = np.array(img_gray)

        label, confidence, all_probs = model_manager.predict(img_array)

        return jsonify({
            'label':      label,
            'thai':       THAI_DIGITS_MAP[label],
            'confidence': round(confidence * 100, 2),
            'all_probs':  [
                {'label': LABELS[i], 'thai': THAI_DIGITS_MAP[LABELS[i]],
                 'prob': round(float(p) * 100, 2)}
                for i, p in enumerate(all_probs)
            ],
            'filename': secure_filename(file.filename),
        })
    except Exception as e:
        logger.error(f"Admin predict error: {e}")
        return jsonify({'error': str(e)}), 500


# ── Admin: add PNG to dataset ──────────────────────────────────────────────────
@app.route('/admin/add-to-dataset', methods=['POST'])
@login_required
def admin_add_to_dataset():
    if 'image_file' not in request.files:
        flash('ไม่พบไฟล์รูปภาพ', 'danger')
        return redirect(url_for('admin_dashboard'))

    file  = request.files['image_file']
    label = request.form.get('label', '').strip()
    split = request.form.get('split', 'train')

    if file.filename == '':
        flash('ไม่ได้เลือกไฟล์', 'danger')
        return redirect(url_for('admin_dashboard'))

    try:
        label_int = int(label)
        if label_int not in LABELS:
            raise ValueError
    except (ValueError, TypeError):
        flash('Label ต้องเป็น 51-55 เท่านั้น', 'danger')
        return redirect(url_for('admin_dashboard'))

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        flash('รองรับเฉพาะ .png, .jpg, .jpeg', 'danger')
        return redirect(url_for('admin_dashboard'))

    dataset_dir = os.path.join(BASE_DIR, '..', 'dataset', split, str(label_int))
    os.makedirs(dataset_dir, exist_ok=True)

    existing  = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_name  = f'{label_int}_custom_{ts}_{len(existing):03d}.png'
    save_path = os.path.join(dataset_dir, new_name)

    img = Image.open(file.stream).convert('L').resize((IMG_SIZE, IMG_SIZE))
    img.save(save_path)

    # Return JSON if called via fetch (XHR), else redirect
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
       'application/json' in request.headers.get('Accept', ''):
        return jsonify({'ok': True, 'saved': new_name})

    flash(f'เพิ่มรูป "{new_name}" เข้า dataset/{split}/{label_int}/ สำเร็จ', 'success')
    return redirect(url_for('admin_dashboard'))


# ── Admin: list saved drawings ─────────────────────────────────────────────────
@app.route('/admin/drawings')
@login_required
def admin_drawings():
    drawings = []
    for fname in sorted(os.listdir(SAVED_DRAWINGS_DIR), reverse=True):
        if fname.lower().endswith('.png'):
            fpath = os.path.join(SAVED_DRAWINGS_DIR, fname)
            drawings.append({
                'name':     fname,
                'size':     os.path.getsize(fpath),
                'modified': datetime.fromtimestamp(
                    os.path.getmtime(fpath)
                ).strftime('%Y-%m-%d %H:%M:%S'),
            })
    return jsonify(drawings)


@app.route('/admin/drawings/delete/<filename>', methods=['POST'])
@login_required
def admin_delete_drawing(filename):
    safe = secure_filename(filename)
    path = os.path.join(SAVED_DRAWINGS_DIR, safe)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({'ok': True})
    return jsonify({'ok': False, 'error': 'not found'}), 404


@app.route('/admin/drawings/image/<filename>')
@login_required
def admin_drawing_image(filename):
    from flask import send_from_directory
    safe = secure_filename(filename)
    return send_from_directory(SAVED_DRAWINGS_DIR, safe)


@app.route('/admin/model-info')
@login_required
def admin_model_info():
    return jsonify(model_manager.get_info())


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
