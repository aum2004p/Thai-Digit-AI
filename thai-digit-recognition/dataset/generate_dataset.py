"""
Generate Thai digit dataset (51-55).
Uses the SAME preprocessing as the web app (crop-to-content → square → resize)
so training images match what the model sees at inference time.

Image pipeline:
  render large (96x96) → crop tight to ink → pad to square → resize 32x32
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import random

THAI_DIGITS = {51: '๕๑', 52: '๕๒', 53: '๕๓', 54: '๕๔', 55: '๕๕'}

IMG_SIZE          = 32
RENDER_SIZE       = 96          # render large, then crop — more detail
SAMPLES_PER_CLASS = 100
TRAIN_RATIO       = 0.8

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

THAI_FONT_PATHS = [
    "C:/Windows/Fonts/THSarabunNew.ttf",
    "C:/Windows/Fonts/THSarabunNew Bold.ttf",
    "C:/Windows/Fonts/cordia.ttf",
    "C:/Windows/Fonts/Cordia New.ttf",
    "C:/Windows/Fonts/CordiaNew.ttf",
    "C:/Windows/Fonts/angsa.ttf",
    "C:/Windows/Fonts/AngsanaNew.ttf",
    "C:/Windows/Fonts/upcil.ttf",
    "C:/Windows/Fonts/Leelawad.ttf",
    "C:/Windows/Fonts/leelawad.ttf",
    "C:/Windows/Fonts/Tahoma.ttf",
    "C:/Windows/Fonts/Arial Unicode MS.ttf",
    "/usr/share/fonts/truetype/thai/Garuda.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/tlwg/Norasi.ttf",
    "/Library/Fonts/Thonburi.ttf",
]


# ── Shared preprocessing (identical to webapp/app.py) ─────────────────────────
def preprocess_for_model(img_gray: Image.Image) -> Image.Image:
    arr = np.array(img_gray, dtype=np.uint8)
    if arr.mean() < 128:
        arr = 255 - arr
    binary = (arr < 200).astype(np.uint8)
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any():
        return Image.fromarray(np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    pad = max(4, int(max(h, w) * 0.15))
    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(arr.shape[1] - 1, cmax + pad)
    cropped = arr[rmin:rmax+1, cmin:cmax+1]
    ch, cw = cropped.shape
    side = max(ch, cw)
    square = np.ones((side, side), dtype=np.uint8) * 255
    top  = (side - ch) // 2
    left = (side - cw) // 2
    square[top:top+ch, left:left+cw] = cropped
    return Image.fromarray(square).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


# ── Font helpers ───────────────────────────────────────────────────────────────
def get_thai_fonts():
    return [f for f in THAI_FONT_PATHS if os.path.exists(f)]


def font_renders_thai(font_path, test_char='๕'):
    try:
        font = ImageFont.truetype(font_path, 24)
        img  = Image.new('L', (48, 48), 255)
        draw = ImageDraw.Draw(img)
        draw.text((4, 4), test_char, fill=0, font=font)
        return bool((np.array(img) < 200).any())
    except Exception:
        return False


# ── Augmentation ───────────────────────────────────────────────────────────────
def augment(img: Image.Image) -> Image.Image:
    """Light augmentation — applied BEFORE preprocessing so crop still works."""
    # Slight rotation
    angle = random.uniform(-18, 18)
    img = img.rotate(angle, fillcolor=255, resample=Image.BICUBIC)
    # Noise
    arr   = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(0, 10), arr.shape)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img   = Image.fromarray(arr)
    # Blur
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.9)))
    # Brightness / contrast
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.4))
    return img


# ── Font-based rendering ───────────────────────────────────────────────────────
def make_font_image(digit_str, font_path):
    """Render digit with font on large canvas, then preprocess."""
    font_size = random.randint(40, 72)
    lw        = random.randint(1, 3)   # slight stroke width via multiple draws

    bg = random.randint(230, 255)
    fg = random.randint(0, 60)

    canvas = RENDER_SIZE * 2
    img    = Image.new('L', (canvas, canvas), bg)
    draw   = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox   = draw.textbbox((0, 0), digit_str, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx     = canvas // 2 - tw // 2
    cy     = canvas // 2 - th // 2

    # Draw with slight offsets to simulate pen weight
    for dx in range(-lw+1, lw):
        for dy in range(-lw+1, lw):
            draw.text((cx+dx, cy+dy), digit_str, fill=fg, font=font)

    img = augment(img)
    return preprocess_for_model(img)


# ── Stroke-based rendering (handwriting simulation) ───────────────────────────
# Strokes defined in normalised [0,1] space for each Thai character
THAI_STROKES = {
    '๑': [
        [(0.50, 0.12), (0.50, 0.88)],
        [(0.34, 0.28), (0.50, 0.12), (0.64, 0.26), (0.52, 0.38)],
    ],
    '๒': [
        [(0.36, 0.24), (0.56, 0.14), (0.70, 0.28), (0.62, 0.44),
         (0.40, 0.58), (0.32, 0.72), (0.34, 0.84), (0.68, 0.86)],
    ],
    '๓': [
        [(0.66, 0.18), (0.44, 0.13), (0.28, 0.28), (0.38, 0.44),
         (0.58, 0.50), (0.68, 0.62), (0.54, 0.76), (0.32, 0.82),
         (0.28, 0.72)],
    ],
    '๔': [
        # เส้นทแยงซ้ายบนลงขวาล่าง
        [(0.65, 0.12), (0.28, 0.55)],
        # เส้นแนวนอนตัดกลาง
        [(0.22, 0.55), (0.72, 0.55)],
        # เส้นตั้งขวา ลงยาว
        [(0.60, 0.12), (0.60, 0.88)],
        # หางล่างขวา
        [(0.48, 0.72), (0.72, 0.72)],
    ],
    '๕': [
        # หัวแบน + ก้านซ้าย
        [(0.68, 0.14), (0.32, 0.14)],
        # ก้านซ้ายลงมา + วนขวา
        [(0.32, 0.14), (0.28, 0.38), (0.48, 0.34),
         (0.68, 0.44), (0.66, 0.62), (0.50, 0.76),
         (0.30, 0.70), (0.24, 0.54)],
    ],
}


def draw_strokes_img(strokes, size, lw, fg, bg):
    img  = Image.new('L', (size, size), bg)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        pts = [(int(x * size), int(y * size)) for x, y in stroke]
        if len(pts) >= 2:
            draw.line(pts, fill=fg, width=lw, joint='curve')
        elif pts:
            r = lw // 2
            x, y = pts[0]
            draw.ellipse([x-r, y-r, x+r, y+r], fill=fg)
    return img


def make_stroke_image(digit_str):
    """Draw Thai digit as vector strokes, then preprocess."""
    lw  = random.randint(3, 7)
    bg  = random.randint(230, 255)
    fg  = random.randint(0, 60)
    scl = random.uniform(0.65, 0.95)   # scale strokes within canvas

    canvas = RENDER_SIZE * 2
    base   = Image.new('L', (canvas, canvas), bg)

    for ch in digit_str:
        strokes = THAI_STROKES.get(ch, [])
        if not strokes:
            continue
        # Scale + centre strokes
        scaled = [
            [(0.5 + (x - 0.5) * scl, 0.5 + (y - 0.5) * scl) for x, y in s]
            for s in strokes
        ]
        ch_img = draw_strokes_img(scaled, canvas, lw, fg, bg)
        arr_b  = np.array(base)
        arr_c  = np.array(ch_img)
        # Composite: take darker pixel (ink)
        base = Image.fromarray(np.minimum(arr_b, arr_c))

    base = augment(base)
    return preprocess_for_model(base)


# ── Main ───────────────────────────────────────────────────────────────────────
def generate_dataset():
    all_fonts  = get_thai_fonts()
    thai_fonts = [f for f in all_fonts if font_renders_thai(f)]
    print(f"Thai-capable fonts: {thai_fonts if thai_fonts else 'none — using strokes only'}")

    for split in ['train', 'test']:
        for label in THAI_DIGITS:
            os.makedirs(os.path.join(OUTPUT_DIR, split, str(label)), exist_ok=True)

    total = 0
    for label, digit_str in THAI_DIGITS.items():
        print(f"Generating {label} ({digit_str})...", end=' ')
        images = []

        for i in range(SAMPLES_PER_CLASS):
            # 60% font-based (if available), 40% stroke-based
            if thai_fonts and random.random() < 0.6:
                img = make_font_image(digit_str, random.choice(thai_fonts))
            else:
                img = make_stroke_image(digit_str)
            images.append(img)

        random.shuffle(images)
        n_train = int(SAMPLES_PER_CLASS * TRAIN_RATIO)

        for idx, img in enumerate(images):
            split     = 'train' if idx < n_train else 'test'
            save_path = os.path.join(OUTPUT_DIR, split, str(label),
                                     f'{label}_{idx:03d}.png')
            img.save(save_path)
            total += 1

        print(f"{n_train} train / {SAMPLES_PER_CLASS - n_train} test")

    print(f"\nDone! Total: {total} images  (32×32 px, preprocessed)")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    generate_dataset()
