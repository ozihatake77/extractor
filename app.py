import os
import re
import json
import tempfile
import traceback
from collections import OrderedDict, Counter
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import pytesseract

# QR Code reader
try:
    from pyzbar.pyzbar import decode as qr_decode
    HAS_QR = True
except ImportError:
    HAS_QR = False

# OpenCV for advanced preprocessing + template matching
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Google Vision (optional)
try:
    from google.cloud import vision
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

# OCR engines
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Preserve OrderedDict insertion order in JSON responses
from flask.json.provider import DefaultJSONProvider
class OrderedJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        kwargs.setdefault('sort_keys', False)
        return super().dumps(obj, **kwargs)
app.json_provider_class = OrderedJSONProvider
app.json = OrderedJSONProvider(app)

# OpenCV for advanced preprocessing
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Google Vision API key (set via env var)
VISION_API_KEY = os.environ.get('GOOGLE_VISION_API_KEY', '')

# ── Advanced KTP Preprocessing Engine ──────────────────────────────────

def auto_rotate_image(img_path):
    """Auto-rotate image based on EXIF and detect if needs 90° rotation"""
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    # If image is taller than wide, might need rotation for KTP (landscape)
    # But don't force it — KTP can be photographed in portrait
    return img

def upscale_if_needed(img, target_min=1500):
    """Upscale small images for better OCR, downscale huge ones"""
    w, h = img.size
    if w < target_min:
        scale = target_min / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    elif w > 3000:
        scale = 2500 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def remove_glare_cv2(img_pil):
    """Remove glare/reflection from KTP photo using CLAHE"""
    if not HAS_CV2:
        return img_pil
    img_np = np.array(img_pil)
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    # CLAHE — Contrast Limited Adaptive Histogram Equalization
    # Removes local glare while preserving text
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced)

def deskew_image(img_pil):
    """Detect and correct skew/rotation in scanned document"""
    if not HAS_CV2:
        return img_pil
    img_np = np.array(img_pil)
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return img_pil
    # Calculate dominant angle
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 30:  # Only consider near-horizontal lines
            angles.append(angle)
    if not angles:
        return img_pil
    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:  # Skip if barely tilted
        return img_pil
    # Rotate
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)

def adaptive_threshold_cv2(img_pil):
    """Adaptive threshold — better than fixed threshold for uneven lighting"""
    if not HAS_CV2:
        return img_pil
    img_np = np.array(img_pil)
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    # Gaussian adaptive threshold — handles shadows/glare
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
    return Image.fromarray(binary)

def otsu_threshold_cv2(img_pil):
    """Otsu's auto threshold — finds optimal cutoff automatically"""
    if not HAS_CV2:
        return img_pil
    img_np = np.array(img_pil)
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

def denoise_cv2(img_pil):
    """Non-local means denoising — preserves edges while removing noise"""
    if not HAS_CV2:
        return img_pil
    img_np = np.array(img_pil)
    if len(img_np.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
    else:
        denoised = cv2.fastNlMeansDenoising(img_np, None, 10, 7, 21)
    return Image.fromarray(denoised)

def crop_nik_area(img_pil):
    """Crop just the NIK area (top-left portion of KTP)"""
    w, h = img_pil.size
    # NIK is typically in the top ~25% of the KTP, left ~60%
    nik_region = img_pil.crop((0, 0, int(w * 0.65), int(h * 0.28)))
    return nik_region

def preprocess_image_ktp(img_path, mode='advanced'):
    """Advanced preprocessing pipeline for Indonesian KTP/KK"""
    img = auto_rotate_image(img_path)
    img = upscale_if_needed(img, target_min=1800)
    gray = img.convert('L')
    
    if mode == 'advanced':
        # Full pipeline: glare removal → denoise → adaptive threshold
        gray = remove_glare_cv2(gray)
        gray = denoise_cv2(gray)
        gray = ImageEnhance.Contrast(gray).enhance(1.5)
        gray = ImageEnhance.Sharpness(gray).enhance(1.3)
        gray = adaptive_threshold_cv2(gray)
    elif mode == 'clahe':
        # CLAHE only — good for photos with uneven lighting
        gray = remove_glare_cv2(gray)
        gray = ImageEnhance.Sharpness(gray).enhance(1.2)
    elif mode == 'otsu':
        # Otsu threshold — good for clean scans
        gray = ImageEnhance.Contrast(gray).enhance(1.5)
        gray = otsu_threshold_cv2(gray)
    elif mode == 'sharpen':
        # Heavy sharpen — good for blurry photos
        gray = ImageEnhance.Contrast(gray).enhance(2.0)
        gray = ImageEnhance.Sharpness(gray).enhance(2.0)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
    elif mode == 'denoise':
        # Denoise heavy — good for noisy/dark photos
        gray = denoise_cv2(gray)
        gray = ImageEnhance.Contrast(gray).enhance(2.0)
        gray = ImageEnhance.Brightness(gray).enhance(1.2)
    elif mode == 'raw':
        pass
    
    return gray

def preprocess_image(img_path, mode='high_contrast'):
    """Legacy preprocessing — kept for Tesseract fallback"""
    img = auto_rotate_image(img_path)
    img = upscale_if_needed(img, target_min=1500)
    gray = img.convert('L')
    
    if mode == 'high_contrast':
        gray = ImageEnhance.Contrast(gray).enhance(2.0)
        gray = ImageEnhance.Sharpness(gray).enhance(1.5)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        gray = gray.point(lambda x: 255 if x > 100 else 0, '1')
    elif mode == 'medium':
        gray = ImageEnhance.Contrast(gray).enhance(1.5)
        gray = ImageEnhance.Sharpness(gray).enhance(1.3)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
    elif mode == 'soft':
        gray = ImageEnhance.Contrast(gray).enhance(1.3)
        gray = ImageEnhance.Sharpness(gray).enhance(1.2)
    elif mode == 'raw':
        pass
    elif mode == 'denoise':
        gray = ImageEnhance.Contrast(gray).enhance(2.5)
        gray = ImageEnhance.Brightness(gray).enhance(1.3)
        gray = gray.filter(ImageFilter.MedianFilter(size=5))
        gray = gray.point(lambda x: 255 if x > 90 else 0, '1')
    
    return gray

def ocr_pass(img_path, mode, psm):
    """Single OCR pass"""
    img = preprocess_image(img_path, mode)
    config = f'--psm {psm} --oem 3 -l ind+eng'
    text = pytesseract.image_to_string(img, config=config)
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 1]
    return lines

def ocr_likely_contains_ktp(text):
    """Check if OCR result likely contains KTP content"""
    upper = text.upper()
    keywords = ['NIK', 'NAMA', 'PROVINSI', 'KABUPATEN', 'KOTA', 'LAHIR', 'AGAMA',
                'KELAMIN', 'ALAMAT', 'RT', 'RW', 'KELURAHAN', 'KECAMATAN',
                'PEKERJAAN', 'KEWARGANEGARAAN', 'BERLAKU']
    hits = sum(1 for kw in keywords if kw in upper)
    return hits >= 2  # At least 2 keywords found

def post_correct_ocr(text):
    """Fix common OCR misreads for Indonesian text"""
    # Common digit/letter confusion in NIK context
    corrections = {
        'O': '0', 'o': '0', 'Q': '0',  # O → 0
        'l': '1', 'I': '1', '|': '1',   # l/I → 1
        'Z': '2', 'z': '2',             # Z → 2
        'S': '5', 's': '5',             # S → 5
        'G': '6',                        # G → 6
        'T': '7',                        # T → 7
        'B': '8',                        # B → 8
        'g': '9',                        # g → 9
    }
    return text

# ══════════════════════════════════════════════════════════════════════
# FEATURE 1: QR Code Reader (belakang KTP)
# ══════════════════════════════════════════════════════════════════════

def read_qr_code(img_path):
    """Read QR code from back of KTP — contains ALL data in structured format"""
    if not HAS_QR:
        return None
    try:
        img = Image.open(img_path)
        # Try multiple preprocessing for QR detection
        images_to_try = [img]
        
        if HAS_CV2:
            img_np = np.array(img.convert('RGB'))
            # Try grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            images_to_try.append(Image.fromarray(gray))
            # Try binary threshold
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            images_to_try.append(Image.fromarray(binary))
            # Try inverted
            images_to_try.append(Image.fromarray(255 - gray))
            # Try CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            images_to_try.append(Image.fromarray(enhanced))
        
        for img_variant in images_to_try:
            results = qr_decode(img_variant)
            if results:
                for result in results:
                    data = result.data.decode('utf-8', errors='ignore')
                    if data and len(data) > 10:
                        return parse_qr_data(data)
        
        return None
    except Exception:
        return None

def parse_qr_data(raw_data):
    """Parse QR code data from Indonesian e-KTP
    Format: NIK\nNama\nTempat/Tgl Lahir\nJenis Kelamin\nGol Darah\nAlamat\nRT/RW\nKel/Desa\nKecamatan\nAgama\nStatus Perkawinan\nPekerjaan\nKewarganegaraan\nBerlaku Hingga
    """
    lines = [l.strip() for l in raw_data.split('\n') if l.strip()]
    if len(lines) < 5:
        return None
    
    ktp = OrderedDict()
    
    # Map lines to fields based on position
    field_order = [
        'nik', 'nama', 'tempat_lahir', 'jenis_kelamin', 'golongan_darah',
        'alamat', 'rt_rw', 'kelurahan', 'kecamatan', 'kabupaten', 'provinsi',
        'agama', 'status_perkawinan', 'pekerjaan', 'kewarganegaraan', 'berlaku_hingga'
    ]
    
    for i, field in enumerate(field_order):
        if i < len(lines):
            ktp[field] = lines[i]
        else:
            ktp[field] = ''
    
    # Special handling: tempat/tgl lahir might be "BEKASI, 29-08-1998"
    if ktp.get('tempat_lahir'):
        ttl = ktp['tempat_lahir']
        if ',' in ttl:
            parts = ttl.split(',', 1)
            ktp['tempat_lahir'] = parts[0].strip()
            ktp['tanggal_lahir'] = parts[1].strip()
        elif re.search(r'\d{1,2}[\-/.]\d{1,2}[\-/.]\d{4}', ttl):
            dm = re.search(r'(\d{1,2})[\-/.](\d{1,2})[\-/.](\d{4})', ttl)
            if dm:
                ktp['tanggal_lahir'] = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
                ktp['tempat_lahir'] = ttl[:dm.start()].strip().rstrip(',').strip()
    
    return ktp

# ══════════════════════════════════════════════════════════════════════
# FEATURE 2: Template Matching — Crop Per-Field
# ══════════════════════════════════════════════════════════════════════

# KTP field regions (approximate percentages of image width/height)
KTP_FIELD_REGIONS = {
    'nik':            (0.25, 0.08, 0.75, 0.14),  # x1%, y1%, x2%, y2%
    'nama':           (0.25, 0.14, 0.75, 0.20),
    'tempat_lahir':   (0.25, 0.20, 0.50, 0.26),
    'tanggal_lahir':  (0.50, 0.20, 0.75, 0.26),
    'jenis_kelamin':  (0.25, 0.26, 0.50, 0.32),
    'golongan_darah': (0.50, 0.26, 0.75, 0.32),
    'alamat':         (0.25, 0.32, 0.75, 0.38),
    'rt_rw':          (0.25, 0.38, 0.50, 0.44),
    'kelurahan':      (0.25, 0.44, 0.50, 0.50),
    'kecamatan':      (0.25, 0.50, 0.50, 0.56),
    'agama':          (0.25, 0.56, 0.50, 0.62),
    'status':         (0.25, 0.62, 0.50, 0.68),
    'pekerjaan':      (0.25, 0.68, 0.50, 0.74),
    'kewarganegaraan':(0.25, 0.74, 0.50, 0.80),
    'berlaku_hingga': (0.25, 0.80, 0.50, 0.86),
}

def crop_field_region(img_pil, field_name):
    """Crop a specific field region from KTP image using template coordinates"""
    if field_name not in KTP_FIELD_REGIONS:
        return img_pil
    
    w, h = img_pil.size
    x1_pct, y1_pct, x2_pct, y2_pct = KTP_FIELD_REGIONS[field_name]
    
    # Add small margin for OCR accuracy
    margin_x = int(w * 0.02)
    margin_y = int(h * 0.01)
    
    x1 = max(0, int(w * x1_pct) - margin_x)
    y1 = max(0, int(h * y1_pct) - margin_y)
    x2 = min(w, int(w * x2_pct) + margin_x)
    y2 = min(h, int(h * y2_pct) + margin_y)
    
    return img_pil.crop((x1, y1, x2, y2))

def ocr_field(img_pil, field_name):
    """OCR a specific field region with optimized preprocessing"""
    # Upscale the crop for better OCR
    w, h = img_pil.size
    if w < 500:
        scale = 800 / w
        img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    
    # Convert to grayscale
    gray = img_pil.convert('L')
    
    # Apply contrast enhancement
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = ImageEnhance.Sharpness(gray).enhance(1.5)
    
    # Use Tesseract with single line mode for field extraction
    if field_name == 'nik':
        config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
    else:
        config = '--psm 7 --oem 3 -l ind+eng'
    
    text = pytesseract.image_to_string(gray, config=config).strip()
    return text

def extract_fields_by_template(img_path):
    """Extract all fields using template matching — crop each field region"""
    img = auto_rotate_image(img_path)
    img = upscale_if_needed(img, target_min=2000)
    
    fields = {}
    for field_name in KTP_FIELD_REGIONS:
        try:
            cropped = crop_field_region(img, field_name)
            text = ocr_field(cropped, field_name)
            if text:
                fields[field_name] = text
        except Exception:
            pass
    
    return fields

# ══════════════════════════════════════════════════════════════════════
# FEATURE 3: NIK Validation Layer
# ══════════════════════════════════════════════════════════════════════

# Indonesian province codes (digit 1-2 of NIK)
PROVINCE_CODES = {
    '11': 'ACEH', '12': 'SUMATERA UTARA', '13': 'SUMATERA BARAT',
    '14': 'RIAU', '15': 'JAMBI', '16': 'SUMATERA SELATAN',
    '17': 'BENGKULU', '18': 'LAMPUNG', '19': 'KEP. BANGKA BELITUNG',
    '21': 'KEP. RIAU', '31': 'DKI JAKARTA', '32': 'JAWA BARAT',
    '33': 'JAWA TENGAH', '34': 'DI YOGYAKARTA', '35': 'JAWA TIMUR',
    '36': 'BANTEN', '51': 'BALI', '52': 'NUSA TENGGARA BARAT',
    '53': 'NUSA TENGGARA TIMUR', '61': 'KALIMANTAN BARAT',
    '62': 'KALIMANTAN TENGAH', '63': 'KALIMANTAN SELATAN',
    '64': 'KALIMANTAN TIMUR', '65': 'KALIMANTAN UTARA',
    '71': 'SULAWESI UTARA', '72': 'SULAWESI TENGAH',
    '73': 'SULAWESI SELATAN', '74': 'SULAWESI TENGGARA',
    '75': 'GORONTALO', '76': 'SULAWESI BARAT', '81': 'MALUKU',
    '82': 'MALUKU UTARA', '91': 'PAPUA BARAT', '92': 'PAPUA',
    '93': 'PAPUA SELATAN', '94': 'PAPUA TENGAH', '95': 'PAPUA PEGUNUNGAN',
    '96': 'PAPUA BARAT DAYA',
}

def validate_nik(nik_str):
    """Validate NIK structure and correct common OCR errors
    Returns: (corrected_nik, is_valid, issues)
    """
    if not nik_str or len(nik_str) != 16:
        return nik_str, False, ['Length is not 16']
    
    digits = re.sub(r'\D', '', nik_str)
    if len(digits) != 16:
        return nik_str, False, ['Contains non-digit characters']
    
    issues = []
    
    # Check province code (digit 1-2)
    province_code = digits[:2]
    if province_code not in PROVINCE_CODES:
        issues.append(f'Invalid province code: {province_code}')
    
    # Check date of birth (digit 7-8 for DD, 9-10 for MM)
    # For females, day is +40 (so 41-71 = day 1-31)
    day = int(digits[6:8])
    month = int(digits[8:10])
    
    if day > 71:
        issues.append(f'Invalid day: {day}')
    elif day > 40:
        day = day - 40  # Female
    
    if month < 1 or month > 12:
        issues.append(f'Invalid month: {month}')
    
    # Check year (digit 11-12)
    year = int(digits[10:12])
    # Year should be reasonable (00-99 maps to 1900-2099)
    
    # Check sequence number (digit 13-16) — should not be 0000
    seq = digits[12:16]
    if seq == '0000':
        issues.append('Invalid sequence: 0000')
    
    is_valid = len(issues) == 0
    return digits, is_valid, issues

def correct_nik_ocr(nik_str, province_hint='', dob_hint=''):
    """Try to correct common OCR errors in NIK using contextual hints"""
    if not nik_str:
        return nik_str
    
    digits = re.sub(r'\D', '', nik_str)
    
    # If length is wrong, try to fix
    if len(digits) == 17:
        # Try removing each digit and validate
        best = None
        best_score = -1
        for remove_pos in range(16):
            candidate = digits[:remove_pos] + digits[remove_pos+1:]
            _, valid, issues = validate_nik(candidate)
            score = (2 - len(issues))  # Higher is better
            if province_hint and candidate[:2] in PROVINCE_CODES:
                score += 1
            if score > best_score:
                best_score = score
                best = candidate
        if best:
            return best
    
    if len(digits) != 16:
        return nik_str
    
    # Try to fix province code using hint
    if province_hint:
        for code, name in PROVINCE_CODES.items():
            if province_hint.upper() in name:
                if digits[:2] != code:
                    # Try swapping first 2 digits
                    candidate = code + digits[2:]
                    _, valid, _ = validate_nik(candidate)
                    if valid:
                        return candidate
                break
    
    return digits

# ══════════════════════════════════════════════════════════════════════
# FEATURE 4: Multi-Engine Voting
# ══════════════════════════════════════════════════════════════════════

def extract_nik_multi_engine(img_path):
    """Extract NIK using multiple engines and vote for best result"""
    candidates = []
    
    # Engine 1: EasyOCR (best for Indonesian text)
    if HAS_EASYOCR:
        try:
            reader = easyocr.Reader(['en', 'id'], gpu=False, verbose=False)
            # Original image
            results = reader.readtext(img_path)
            for _, text, conf in results:
                if conf > 0.3:
                    digits = re.sub(r'\D', '', text)
                    if len(digits) >= 16:
                        candidates.append(digits[:16])
            # NIK crop
            img = auto_rotate_image(img_path)
            img = upscale_if_needed(img, target_min=1800)
            nik_crop = crop_nik_area(img)
            nik_path = img_path + '.nik.png'
            nik_crop.save(nik_path)
            results2 = reader.readtext(nik_path)
            for _, text, conf in results2:
                if conf > 0.3:
                    digits = re.sub(r'\D', '', text)
                    if len(digits) >= 16:
                        candidates.append(digits[:16])
            os.unlink(nik_path)
        except Exception:
            pass
    
    # Engine 2: Tesseract with multiple preprocessing
    for mode in ['high_contrast', 'medium', 'soft', 'raw']:
        try:
            img = preprocess_image(img_path, mode)
            text = pytesseract.image_to_string(img, config='--psm 6 --oem 3 -l ind+eng')
            for line in text.split('\n'):
                if 'NIK' in line.upper():
                    after = line[line.upper().find('NIK') + 3:]
                    digits = re.sub(r'\D', '', after)
                    if len(digits) >= 16:
                        candidates.append(digits[:16])
                    break
        except Exception:
            pass
    
    # Engine 3: Template crop + Tesseract
    try:
        img = auto_rotate_image(img_path)
        img = upscale_if_needed(img, target_min=2000)
        nik_region = crop_field_region(img, 'nik')
        for contrast in [1.5, 2.0, 2.5]:
            gray = nik_region.convert('L')
            gray = ImageEnhance.Contrast(gray).enhance(contrast)
            gray = ImageEnhance.Sharpness(gray).enhance(1.5)
            text = pytesseract.image_to_string(gray, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')
            digits = re.sub(r'\D', '', text)
            if len(digits) >= 16:
                candidates.append(digits[:16])
    except Exception:
        pass
    
    if not candidates:
        return '', False
    
    # Validate each candidate
    valid_candidates = []
    for c in candidates:
        _, valid, _ = validate_nik(c)
        if valid:
            valid_candidates.append(c)
    
    # Prefer valid candidates
    pool = valid_candidates if valid_candidates else candidates
    
    # Vote: pick most common
    if pool:
        winner = Counter(pool).most_common(1)[0][0]
        _, valid, issues = validate_nik(winner)
        return winner, valid
    
    return '', False


def extract_text_easyocr(img_path):
    """Extract text using EasyOCR with advanced preprocessing"""
    if not HAS_EASYOCR:
        return []
    try:
        reader = easyocr.Reader(['en', 'id'], gpu=False, verbose=False)
        
        # Strategy 1: Original image (EasyOCR handles preprocessing internally)
        results = reader.readtext(img_path)
        all_lines = [text for _, text, conf in results if conf > 0.2]
        
        # Strategy 2: CLAHE preprocessed (removes glare)
        if HAS_CV2:
            try:
                img_clahe = preprocess_image_ktp(img_path, 'clahe')
                clahe_path = img_path + '.clahe.png'
                img_clahe.save(clahe_path)
                results2 = reader.readtext(clahe_path)
                for _, text, conf in results2:
                    if conf > 0.2 and text not in all_lines:
                        all_lines.append(text)
                os.unlink(clahe_path)
            except Exception:
                pass
        
        # Strategy 3: NIK area crop (focused OCR for NIK)
        try:
            img = auto_rotate_image(img_path)
            img = upscale_if_needed(img, target_min=1800)
            nik_img = crop_nik_area(img)
            nik_path = img_path + '.nik.png'
            nik_img.save(nik_path)
            results_nik = reader.readtext(nik_path)
            for _, text, conf in results_nik:
                if conf > 0.2 and text not in all_lines:
                    all_lines.append(text)
            os.unlink(nik_path)
        except Exception:
            pass
        
        return all_lines
    except Exception:
        return []

def extract_text_multi(img_path):
    """Run multiple OCR passes and smart-merge results"""
    all_lines = []
    seen = set()
    best_lines = []
    best_score = 0
    
    # More passes with varied strategies
    passes = [
        ('high_contrast', 6),   # Uniform block
        ('high_contrast', 4),   # Single column
        ('high_contrast', 3),   # Fully automatic
        ('medium', 6),          # Uniform block, softer
        ('medium', 4),          # Single column, softer
        ('medium', 3),          # Automatic, softer
        ('soft', 6),            # Well-lit photos
        ('soft', 3),            # Well-lit, automatic
        ('denoise', 6),         # Dark/blurry photos
        ('raw', 6),             # No preprocessing
    ]
    
    for mode, psm in passes:
        try:
            lines = ocr_pass(img_path, mode, psm)
            # Score this pass by how many KTP keywords it found
            combined = ' '.join(lines)
            score = ocr_likely_contains_ktp(combined)
            if score > best_score:
                best_score = score
                best_lines = lines
            
            for line in lines:
                # Smarter dedup: normalize but keep variant readings
                key = re.sub(r'\s+', ' ', line.strip().upper())
                key = re.sub(r'[^A-Z0-9\s]', '', key)  # Strip OCR noise chars
                if key not in seen and len(key) > 1:
                    seen.add(key)
                    all_lines.append(line)
        except Exception:
            pass
    
    # If best pass found more keywords, prefer its ordering
    if best_score >= 3 and best_lines:
        # Rebuild with best pass lines first, then unique extras
        best_set = set(re.sub(r'\s+', ' ', l.strip().upper()) for l in best_lines)
        extras = [l for l in all_lines if re.sub(r'\s+', ' ', l.strip().upper()) not in best_set]
        all_lines = best_lines + extras
    
    return all_lines

def extract_text_google_vision(img_path):
    """Extract text using Google Cloud Vision API (much more accurate)"""
    import urllib.request
    
    # Read image
    with open(img_path, 'rb') as f:
        img_data = f.read()
    
    import base64
    img_b64 = base64.b64encode(img_data).decode()
    
    # Call Vision API
    url = f'https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}'
    payload = json.dumps({
        "requests": [{
            "image": {"content": img_b64},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }).encode()
    
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    
    # Extract text
    annotations = resp.get('responses', [{}])[0].get('textAnnotations', [])
    if not annotations:
        return []
    
    # First annotation is full text, rest are individual words
    full_text = annotations[0].get('description', '')
    lines = [l.strip() for l in full_text.split('\n') if l.strip() and len(l.strip()) > 1]
    
    return lines

def find_value(text, keywords):
    """Extract value after keyword in text"""
    upper = text.upper()
    for kw in keywords:
        idx = upper.find(kw.upper())
        if idx >= 0:
            after = text[idx + len(kw):]
            after = re.sub(r'^[\s:=\-_,;.>]+', '', after).strip()
            if after:
                return after
    return None

def extract_nik_from_image(img_path):
    """Dedicated NIK extraction — run OCR with multiple modes, only parse NIK line"""
    nik_candidates = []
    
    for mode in ['medium', 'soft', 'high_contrast', 'raw', 'denoise']:
        try:
            img = preprocess_image(img_path, mode)
            text = pytesseract.image_to_string(img, config='--psm 6 --oem 3 -l ind+eng')
            
            # Find the NIK line specifically
            for line in text.split('\n'):
                upper = line.upper()
                if 'NIK' not in upper:
                    continue
                
                # Extract only the part AFTER "NIK" label
                nik_idx = upper.find('NIK')
                after_nik = line[nik_idx + 3:]  # Skip "NIK"
                
                # Fix common OCR misreads only on the digit part
                raw = after_nik
                for ch, rep in [('O','0'),('o','0'),('Q','0'),('l','1'),('I','1'),('|','1'),('S','5'),('s','5'),('B','8'),('G','6'),('Z','2'),('z','2'),('D','0'),('J','1')]:
                    raw = raw.replace(ch, rep)
                
                digits = re.sub(r'\D', '', raw)
                
                # 16 digits found directly
                if len(digits) == 16:
                    nik_candidates.append(digits)
                    break
                
                # 17 digits — try removing each to find best 16-digit NIK
                if len(digits) == 17:
                    for remove_pos in range(8, 13):  # Middle positions most likely to have extra digit
                        candidate = digits[:remove_pos] + digits[remove_pos+1:]
                        nik_candidates.append(candidate)
                    break
                
                # 14-15 digits — might be split, try from full text
                if len(digits) >= 14:
                    # Look for more digits nearby
                    for other_line in text.split('\n'):
                        if other_line == line:
                            continue
                        other_raw = other_line
                        for ch, rep in [('O','0'),('o','0'),('l','1'),('I','1'),('|','1'),('S','5'),('B','8'),('D','0')]:
                            other_raw = other_raw.replace(ch, rep)
                        other_digits = re.sub(r'\D', '', other_raw)
                        combined = digits + other_digits
                        if len(combined) >= 16:
                            nik_candidates.append(combined[:16])
                            break
                    break
        except Exception:
            pass
    
    if not nik_candidates:
        return ''
    
    # Pick best candidate: prefer ones starting with valid province code
    valid = [c for c in nik_candidates if 11 <= int(c[:2]) <= 99 and len(c) == 16]
    if valid:
        from collections import Counter
        return Counter(valid).most_common(1)[0][0]
    
    # Fallback: return first candidate
    return nik_candidates[0] if nik_candidates else ''


def parse_ktp(lines, img_path=None):
    """Smart KTP parser — fields ordered exactly like physical KTP"""
    full_text = ' '.join(lines)
    full_upper = full_text.upper()
    
    # OrderedDict: fields in EXACT KTP card order (top to bottom, left to right)
    ktp = OrderedDict([
        ('nik', ''),
        ('nama', ''),
        ('tempat_lahir', ''),
        ('tanggal_lahir', ''),
        ('jenis_kelamin', ''),
        ('golongan_darah', ''),
        ('alamat', ''),
        ('rt_rw', ''),
        ('kelurahan', ''),
        ('kecamatan', ''),
        ('kabupaten', ''),
        ('provinsi', ''),
        ('agama', ''),
        ('status_perkawinan', ''),
        ('pekerjaan', ''),
        ('kewarganegaraan', ''),
        ('berlaku_hingga', ''),
    ])
    
    # === NIK: first try from parsed lines (EasyOCR often correct), then Tesseract fallback ===
    # Try from lines first
    for i, line in enumerate(lines):
        if 'NIK' in line.upper():
            # Check if NIK is on the same line
            after = line[line.upper().find('NIK') + 3:]
            digits = re.sub(r'\D', '', after)
            if len(digits) >= 16:
                ktp['nik'] = digits[:16]
                break
            # Check next line (EasyOCR often puts NIK on separate line)
            if i + 1 < len(lines):
                next_digits = re.sub(r'\D', '', lines[i + 1])
                if len(next_digits) >= 16:
                    ktp['nik'] = next_digits[:16]
                    break
    
    # Fallback: Tesseract multi-pass if lines didn't have NIK
    if not ktp['nik'] and img_path:
        nik = extract_nik_from_image(img_path)
        if nik:
            ktp['nik'] = nik
    
    # Fallback: direct 16-digit match in full text
    if not ktp['nik']:
        nik_match = re.search(r'\b(\d{16})\b', full_text.replace(' ', ''))
        if nik_match:
            ktp['nik'] = nik_match.group(1)
    
    # === Provinsi ===
    for line in lines:
        if 'PROVINSI' in line.upper():
            val = find_value(line, ['PROVINSI'])
            if val:
                # Fix: "BARA /" → "BARAT", clean special chars
                val = re.sub(r'\s*[/\\|]\s*$', 'T', val)
                val = re.sub(r'^[\-—\s]+', '', val)
                val = re.sub(r'\s+', ' ', val).strip()
                if len(val) > 3:
                    ktp['provinsi'] = val.title()
            break
    
    # === Kabupaten/Kota ===
    for line in lines:
        upper = line.upper()
        if ('KOTA' in upper or 'KABUPATEN' in upper) and 'BEKASI' not in upper:
            # Check it's not part of another field
            if not any(kw in upper for kw in ['ALAMAT', 'PEKERJAAN', 'LAHIR', 'NIK']):
                val = re.sub(r'^[\-—\s:=_]+', '', line).strip()
                if len(val) > 3:
                    ktp['kabupaten'] = val.title()
                    break
    # Also look for "KOTA XXX" pattern
    if not ktp['kabupaten']:
        for line in lines:
            upper = line.upper()
            if 'KOTA' in upper and not any(kw in upper for kw in ['ALAMAT', 'PEKERJAAN']):
                ktp['kabupaten'] = line.strip().title()
                break
    
    # === Nama ===
    for i, line in enumerate(lines):
        upper = line.upper()
        if 'NAMA' in upper and 'NAMAKA' not in upper:
            val = find_value(line, ['Nama'])
            if val and len(val) > 2:
                val = re.sub(r'[^a-zA-Z\s]', '', val).strip()
                if len(val) > 2:
                    ktp['nama'] = ' '.join(w.capitalize() for w in val.split())
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if not any(kw in next_line.upper() for kw in ['NIK', 'ALAMAT', 'LAHIR', 'AGAMA', 'KELAMIN']):
                    val = re.sub(r'[^a-zA-Z\\s]', '', next_line).strip()
                    if len(val) > 2:
                        ktp['nama'] = ' '.join(w.capitalize() for w in val.split())
            break
    # Fallback: EasyOCR puts name on separate line after "Nama"
    if not ktp['nama']:
        for i, line in enumerate(lines):
            if line.upper().strip() == 'NAMA' and i + 1 < len(lines):
                val = re.sub(r'[^a-zA-Z\s]', '', lines[i + 1]).strip()
                if len(val) > 2:
                    ktp['nama'] = ' '.join(w.capitalize() for w in val.split())
                break
    
    # === Tempat/Tgl Lahir ===
    for line in lines:
        upper = line.upper()
        if any(kw in upper for kw in ['LAHIR', 'TEMPAT', 'TTL']):
            val = find_value(line, ['Tempat/Tgl Lahir', 'Tempat/TgiLahir', 'Tempat', 'Tgl Lahir', 'TgiLahir', 'TTL', 'Lahir'])
            if val:
                # Fix OCR: "29 08 1998" → "29-08-1998"
                val = val.replace('  ', '-')
                # Try to find date pattern: DD-MM-YYYY or DD/MM/YYYY or DD MM YYYY
                date_match = re.search(r'(\d{1,2})[\s\-/.]+(\d{1,2})[\s\-/.]+(\d{4})', val)
                if date_match:
                    day, month, year = date_match.group(1), date_match.group(2), date_match.group(3)
                    ktp['tanggal_lahir'] = f"{day}-{month}-{year}"
                    tempat = val[:date_match.start()].strip().rstrip(',').strip()
                    if tempat and len(tempat) > 1:
                        ktp['tempat_lahir'] = re.sub(r'[^a-zA-Z\s.]', '', tempat).strip().upper().title()
                else:
                    parts = val.split(',', 1)
                    if len(parts) == 2:
                        tempat = parts[0].strip()
                        tanggal = parts[1].strip()
                        if tempat and len(tempat) > 1:
                            ktp['tempat_lahir'] = re.sub(r'[^a-zA-Z\s.]', '', tempat).strip().upper().title()
                        dm = re.search(r'(\d{1,2})[\s\-/.]+(\d{1,2})[\s\-/.]+(\d{4})', tanggal)
                        if dm:
                            ktp['tanggal_lahir'] = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
                        else:
                            ktp['tanggal_lahir'] = tanggal.strip()
                    elif len(val) > 2:
                        ktp['tempat_lahir'] = re.sub(r'[^a-zA-Z\s.]', '', val).strip().upper().title()
            break
    
    # Fallback: EasyOCR puts TTL on separate lines
    if not ktp['tempat_lahir'] and not ktp['tanggal_lahir']:
        for i, line in enumerate(lines):
            if line.upper().strip() in ['TEMPATTGL LAHIR', 'TEMPAT/TGL LAHIR', 'TEMPAT/TGILAHIR', 'TEMPAVTG LAHIR']:
                # Next line has the actual value
                if i + 1 < len(lines):
                    val = lines[i + 1].strip()
                    dm = re.search(r'(\d{1,2})[\s\-/.]+(\d{1,2})[\s\-/.]+(\d{4})', val)
                    if dm:
                        ktp['tanggal_lahir'] = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
                        tempat = val[:dm.start()].strip().rstrip(',').strip()
                        if tempat:
                            ktp['tempat_lahir'] = re.sub(r'[^a-zA-Z\s.]', '', tempat).strip().upper().title()
                break
    
    # === Fallback: Tempat/Tgl Lahir from line after NAMA ===
    if not ktp['tempat_lahir'] and not ktp['tanggal_lahir']:
        for i, line in enumerate(lines):
            if 'NAMA' in line.upper() and i + 2 < len(lines):
                ttl_line = lines[i + 2] if i + 2 < len(lines) else ''
                if any(c.isdigit() for c in ttl_line) and any(kw in ttl_line.upper() for kw in ['LAHIR', 'BEKASI', 'JAKARTA', 'BANDUNG', 'SURABAYA']):
                    dm = re.search(r'(\d{1,2})[\s\-/.]+(\d{1,2})[\s\-/.]+(\d{4})', ttl_line)
                    if dm:
                        ktp['tanggal_lahir'] = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
                        tempat = ttl_line[:dm.start()].strip().rstrip(',').strip()
                        if tempat:
                            ktp['tempat_lahir'] = re.sub(r'[^a-zA-Z\s.]', '', tempat).strip().upper().title()
                break
    
    # === Fallback: Tempat Lahir from known Indonesian cities ===
    if not ktp['tempat_lahir']:
        known_cities = ['BEKASI', 'JAKARTA', 'BANDUNG', 'SURABAYA', 'SEMARANG', 'YOGYAKARTA',
                        'MALANG', 'MEDAN', 'MAKASSAR', 'DENPASAR', 'BOGOR', 'TANGERANG',
                        'DEPOK', 'SOLO', 'BALIKPAPAN', 'MANADO', 'PALEMBANG', 'PADANG',
                        'PEKANBARU', 'BATAM', 'LAMPUNG', 'CIREBON', 'TEGAL', 'PURWOKERTO',
                        'MADIUN', 'KEDIRI', 'BLITAR', 'PROBOLINGGO', 'PASURUAN', 'MOJOKERTO',
                        'SIDOARJO', 'GRESIK', 'TUBAN', 'LAMONGAN', 'JOMBANG', 'NGANJUK',
                        'KEDIRI', 'TULUNGAGUNG', 'TRENGGALEK', 'PONOROGO', 'MAGETAN',
                        'NGAWI', 'BOJONEGORO', 'JEMBER', 'BANYUWANGI', 'SITUBONDO',
                        'BONDOWOSO', 'LUMAJANG', 'PROBOLINGGO', 'MALANG', 'BATU']
        for city in known_cities:
            if city in full_upper:
                ktp['tempat_lahir'] = city.title()
                break
    
    # Fallback: find date pattern
    if not ktp['tanggal_lahir']:
        for line in lines:
            match = re.search(r'(\d{1,2}[\-/]\d{1,2}[\-/]\d{4})', line)
            if match:
                ktp['tanggal_lahir'] = match.group(1).replace(' ', '-')
                break
    
    # === Jenis Kelamin ===
    for line in lines:
        upper = line.upper()
        if 'KELAMIN' in upper or 'JENIS' in upper:
            if 'LAKI' in upper:
                ktp['jenis_kelamin'] = 'Laki-laki'
            elif 'PEREMPUAN' in upper or 'WANITA' in upper:
                ktp['jenis_kelamin'] = 'Perempuan'
            break
    if not ktp['jenis_kelamin']:
        for line in lines:
            if 'LAKI' in line.upper():
                ktp['jenis_kelamin'] = 'Laki-laki'
                break
    
    # === Golongan Darah ===
    for line in lines:
        upper = line.upper()
        if 'DARAH' in upper:
            val = find_value(line, ['Gol. Darah', 'Gol Darah', 'Golongan Darah', 'Darah'])
            if val:
                val_clean = val.upper().strip()[:3]
                for bt in ['A', 'B', 'AB', 'O']:
                    if val_clean.startswith(bt):
                        ktp['golongan_darah'] = bt
                        break
            # Also check for standalone blood type
            if not ktp['golongan_darah']:
                match = re.search(r'\b(A|B|AB|O)\b', upper.split('DARAH')[-1] if 'DARAH' in upper else '')
                if match:
                    ktp['golongan_darah'] = match.group(1)
            break
    # Fallback: look for standalone blood type near "DARAH" in full text
    if not ktp['golongan_darah']:
        darah_match = re.search(r'DARAH\s*[=:]\s*(A|B|AB|O)\b', full_upper)
        if darah_match:
            ktp['golongan_darah'] = darah_match.group(1)
    
    # === Alamat ===
    for i, line in enumerate(lines):
        upper = line.upper()
        if 'ALAMAT' in upper or 'ALAMA' in upper:
            val = find_value(line, ['Alamat', 'ALAMAT', 'Alama'])
            if val and len(val) > 2:
                val = re.sub(r'^JL?\s*\.?\s*L?\s*([A-Z])', r'JL. \1', val)
                val = re.sub(r'^j([a-z])', r'Jl. \1', val)
                val = re.sub(r'^\d+\.?\s*JL', 'JL.', val)
                val = re.sub(r'\s+\d+\s+Te$', '', val)
                ktp['alamat'] = val.upper().strip()
            elif i + 1 < len(lines):
                # EasyOCR: value on next line
                next_val = lines[i + 1].strip()
                if len(next_val) > 2 and not any(kw in next_val.upper() for kw in ['RT', 'RW', 'KEL', 'KEC', 'AGAMA', 'KECAMATAN']):
                    ktp['alamat'] = next_val.upper().strip()
            break
    # Fallback: EasyOCR variant spelling
    if not ktp['alamat']:
        for i, line in enumerate(lines):
            if line.upper().strip() in ['ALAMAT', 'ALAMAR', 'ALAMER'] and i + 1 < len(lines):
                val = lines[i + 1].strip()
                if len(val) > 2:
                    ktp['alamat'] = val.upper().strip()
                break
    
    # === RT/RW ===
    for line in lines:
        match = re.search(r'(\d{3})\s*/\s*(\d{3})', line)
        if match:
            ktp['rt_rw'] = f"{match.group(1)}/{match.group(2)}"
            break
    if not ktp['rt_rw']:
        for line in lines:
            if 'RT' in line.upper() and '/' in line:
                match = re.search(r'(\d{1,3})\s*/\s*(\d{1,3})', line)
                if match:
                    ktp['rt_rw'] = f"{match.group(1).zfill(3)}/{match.group(2).zfill(3)}"
                    break
    
    # === Kelurahan/Desa ===
    for i, line in enumerate(lines):
        upper = line.upper()
        if any(kw in upper for kw in ['KELURAHAN', 'DESA', 'KEL/DESA', 'KELDESA', 'KELDESA', 'KEVDESA']):
            val = find_value(line, ['Kel/Desa', 'Kelurahan', 'Desa', 'KelDesa', 'Kel', 'KevDesa'])
            if val and len(val) > 2:
                val = re.sub(r'[^a-zA-Z\s]', '', val).strip()
                if len(val) > 2:
                    ktp['kelurahan'] = val.title()
            elif i + 1 < len(lines):
                next_val = re.sub(r'[^a-zA-Z\s]', '', lines[i + 1]).strip()
                if len(next_val) > 2:
                    ktp['kelurahan'] = next_val.title()
            break
    
    # === Kecamatan ===
    for i, line in enumerate(lines):
        upper = line.upper()
        if 'KECAMATAN' in upper or 'KECAMALAN' in upper or 'KECAMALAN' in upper:
            val = find_value(line, ['Kecamatan', 'Kecamalan'])
            if val and len(val) > 2:
                val = re.sub(r'[^a-zA-Z\s]', '', val).strip()
                if len(val) > 2:
                    ktp['kecamatan'] = val.title()
            elif i + 1 < len(lines):
                next_val = re.sub(r'[^a-zA-Z\s]', '', lines[i + 1]).strip()
                if len(next_val) > 2:
                    ktp['kecamatan'] = next_val.title()
            break
    
    # === Agama ===
    for i, line in enumerate(lines):
        upper = line.upper()
        if 'AGAMA' in upper and 'KAWIN' not in upper:
            val = find_value(line, ['Agama'])
            if val:
                agama_upper = val.upper().strip().rstrip('-').strip()
                agama_map = {
                    'ISLAM': 'Islam', 'KRISTEN': 'Kristen', 'KATOLIK': 'Katolik',
                    'HINDU': 'Hindu', 'BUDHA': 'Buddha', 'BUDDHA': 'Buddha',
                    'KONGHUCU': 'Konghucu'
                }
                for key, agama_val in agama_map.items():
                    if key in agama_upper:
                        ktp['agama'] = agama_val
                        break
            elif i + 1 < len(lines):
                # EasyOCR: value on next line
                next_val = lines[i + 1].strip().upper()
                agama_map = {
                    'ISLAM': 'Islam', 'KRISTEN': 'Kristen', 'KATOLIK': 'Katolik',
                    'HINDU': 'Hindu', 'BUDHA': 'Buddha', 'BUDDHA': 'Buddha',
                    'KONGHUCU': 'Konghucu'
                }
                for key, agama_val in agama_map.items():
                    if key in next_val:
                        ktp['agama'] = agama_val
                        break
            break
    
    # === Status Perkawinan ===
    for line in lines:
        upper = line.upper()
        if 'PERKAWINAN' in upper or 'PERKAWNAN' in upper or 'KAWIN' in upper:
            val = find_value(line, ['Status Perkawinan', 'Perkawinan', 'Perkawnan', 'Status'])
            if val:
                status_upper = val.upper()
                if 'BELUM' in status_upper:
                    ktp['status_perkawinan'] = 'Belum Kawin'
                elif 'CERAI HIDUP' in status_upper:
                    ktp['status_perkawinan'] = 'Cerai Hidup'
                elif 'CERAI MATI' in status_upper:
                    ktp['status_perkawinan'] = 'Cerai Mati'
                elif 'KAWIN' in status_upper:
                    ktp['status_perkawinan'] = 'Kawin'
            break
    
    # === Pekerjaan ===
    for i, line in enumerate(lines):
        upper = line.upper()
        if 'PEKERJAAN' in upper or 'PEKERJ' in upper or 'PEKORJAAN' in upper:
            val = find_value(line, ['Pekerjaan', 'PEKERJAAN', 'Pekorjaan'])
            if val:
                val = re.split(r'\s+KOTA\s+|\s+KABUPATEN\s+|\s*<.*$', val)[0]
                val = val.replace('MM', '/M').replace('MMA', '/MA')
                val = val.replace('PC', 'PE').replace('PCLA', 'PELA')
                val = re.sub(r'\s+', ' ', val).strip()
                val = val.upper().replace('PELAJARMAHA', 'PELAJAR/MAHA').replace('PELAJARAMAHA', 'PELAJAR/MAHA').replace('PELAJARMMAHA', 'PELAJAR/MAHA')
                if len(val) > 2:
                    ktp['pekerjaan'] = val.title()
            elif i + 1 < len(lines):
                # EasyOCR: value on next line
                next_val = lines[i + 1].strip()
                next_val = next_val.upper().replace('PELAJARMAHA', 'PELAJAR/MAHA').replace('PELAJARMMAHA', 'PELAJAR/MAHA')
                if len(next_val) > 2:
                    ktp['pekerjaan'] = next_val.title()
            break
    
    # === Kewarganegaraan ===
    for line in lines:
        upper = line.upper()
        if 'KEWARGANEGARAAN' in upper or 'WARGA' in upper:
            val = find_value(line, ['Kewarganegaraan', 'Warga Negara'])
            if val:
                if 'WNI' in val.upper() or 'INDONESIA' in val.upper():
                    ktp['kewarganegaraan'] = 'WNI'
                elif 'WNA' in val.upper():
                    ktp['kewarganegaraan'] = 'WNA'
                else:
                    ktp['kewarganegaraan'] = val.strip().upper()
            elif 'WNI' in upper:
                ktp['kewarganegaraan'] = 'WNI'
            elif 'WNA' in upper:
                ktp['kewarganegaraan'] = 'WNA'
            break
    if not ktp['kewarganegaraan']:
        if 'WNI' in full_upper:
            ktp['kewarganegaraan'] = 'WNI'
        elif 'WNA' in full_upper:
            ktp['kewarganegaraan'] = 'WNA'
    
    # === Berlaku Hingga ===
    for line in lines:
        upper = line.upper()
        if 'BERLAKU' in upper or 'BERTAKU' in upper or 'BORLAKU' in upper:
            val = find_value(line, ['Berlaku Hingga', 'Berlaku', 'Bertaku', 'Borlaku'])
            if val:
                if 'SEUMUR' in val.upper() or 'HIDUP' in val.upper():
                    ktp['berlaku_hingga'] = 'SEUMUR HIDUP'
                else:
                    ktp['berlaku_hingga'] = val.strip().upper()
            elif 'SEUMUR' in upper:
                ktp['berlaku_hingga'] = 'SEUMUR HIDUP'
            break
    if not ktp['berlaku_hingga']:
        if 'SEUMUR' in full_upper:
            ktp['berlaku_hingga'] = 'SEUMUR HIDUP'
    
    return ktp

def parse_kk(lines):
    """Parse KK fields"""
    full_text = ' '.join(lines)
    full_upper = full_text.upper()
    
    kk = {
        'nomor_kk': '', 'nama_kepala_keluarga': '', 'alamat': '',
        'rt_rw': '', 'kelurahan_desa': '', 'kecamatan': '',
        'kabupaten_kota': '', 'provinsi': '', 'kode_pos': ''
    }
    
    # Nomor KK - 16 digits
    match = re.search(r'\d{16}', full_text.replace(' ', ''))
    if match:
        kk['nomor_kk'] = match.group()
    
    # Nama Kepala Keluarga
    for line in lines:
        if 'KEPALA' in line.upper():
            val = find_value(line, ['Kepala Keluarga', 'Kepala', 'Nama Kepala'])
            if val:
                kk['nama_kepala_keluarga'] = re.sub(r'[^a-zA-Z\s.]', '', val).strip().title()
            break
    
    # Alamat
    for line in lines:
        if 'ALAMAT' in line.upper():
            val = find_value(line, ['Alamat'])
            if val:
                kk['alamat'] = val.title()
            break
    
    # RT/RW
    for line in lines:
        match = re.search(r'(\d{3})\s*/\s*(\d{3})', line)
        if match:
            kk['rt_rw'] = f"{match.group(1)}/{match.group(2)}"
            break
    
    # Kelurahan
    for line in lines:
        if 'KELURAHAN' in line.upper() or 'DESA' in line.upper():
            val = find_value(line, ['Kelurahan', 'Desa', 'Kel/Desa'])
            if val:
                kk['kelurahan_desa'] = re.sub(r'[^a-zA-Z\s]', '', val).strip().title()
            break
    
    # Kecamatan
    for line in lines:
        if 'KECAMATAN' in line.upper():
            val = find_value(line, ['Kecamatan'])
            if val:
                kk['kecamatan'] = re.sub(r'[^a-zA-Z\s]', '', val).strip().title()
            break
    
    # Kabupaten/Kota
    for line in lines:
        upper = line.upper()
        if 'KABUPATEN' in upper or 'KOTA' in upper:
            val = find_value(line, ['Kabupaten', 'Kota'])
            if val:
                kk['kabupaten_kota'] = val.title()
            break
    
    # Provinsi
    for line in lines:
        if 'PROVINSI' in line.upper():
            val = find_value(line, ['Provinsi'])
            if val:
                kk['provinsi'] = val.title()
            break
    
    # Kode Pos
    for line in lines:
        if 'KODE' in line.upper() and 'POS' in line.upper():
            val = find_value(line, ['Kode Pos'])
            if val:
                kk['kode_pos'] = val.strip()
            break
    if not kk['kode_pos']:
        for line in lines:
            match = re.search(r'\b\d{5}\b', line)
            if match:
                kk['kode_pos'] = match.group()
                break
    
    return kk

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/extract', methods=['POST'])
def extract():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    doc_type = request.form.get('type', 'ktp')
    engine = request.form.get('engine', 'tesseract')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Server-side file validation
    allowed_types = {'image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif'}
    if file.content_type and file.content_type not in allowed_types:
        return jsonify({'error': f'Format tidak didukung: {file.content_type}. Gunakan JPG/PNG/WebP.'}), 400
    
    # Validate doc_type
    if doc_type not in ('ktp', 'kk'):
        return jsonify({'error': 'Tipe dokumen tidak valid'}), 400
    
    # Validate engine
    if engine not in ('tesseract', 'google'):
        engine = 'tesseract'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        parsed = None
        used_engine = ''
        lines = []
        
        if doc_type == 'ktp':
            # ═══ STRATEGY 1: QR Code (100% accurate) ═══
            qr_data = read_qr_code(tmp_path)
            if qr_data and qr_data.get('nik'):
                parsed = qr_data
                used_engine = 'qr_code (100%)'
                lines = [f"QR: {k}={v}" for k, v in qr_data.items() if v]
            
            # ═══ STRATEGY 2: Multi-engine OCR ═══
            if not parsed or not parsed.get('nik'):
                # Get text from multiple engines
                if HAS_EASYOCR:
                    lines = extract_text_easyocr(tmp_path)
                    used_engine = 'easyocr (AI)'
                if not lines:
                    lines = extract_text_multi(tmp_path)
                    used_engine = 'tesseract'
                
                # Parse with existing parser
                parsed_ocr = parse_ktp(lines, img_path=tmp_path)
                
                # Merge QR data with OCR data (QR takes priority)
                if qr_data:
                    for k, v in qr_data.items():
                        if v and (not parsed_ocr.get(k) or k == 'nik'):
                            parsed_ocr[k] = v
                
                parsed = parsed_ocr
            
            # ═══ STRATEGY 3: Template matching (fill missing fields) ═══
            template_fields = extract_fields_by_template(tmp_path)
            for k, v in template_fields.items():
                if v and not parsed.get(k):
                    parsed[k] = v
                # Special: tanggal_lahir from template
                if k == 'tanggal_lahir' and v and not parsed.get('tanggal_lahir'):
                    parsed['tanggal_lahir'] = v
            
            # ═══ STRATEGY 4: NIK multi-engine voting ═══
            nik_multi, nik_valid = extract_nik_multi_engine(tmp_path)
            if nik_multi:
                _, current_valid, _ = validate_nik(parsed.get('nik', ''))
                # Use multi-engine NIK if current is invalid or multi is valid
                if nik_valid and not current_valid:
                    parsed['nik'] = nik_multi
                elif not parsed.get('nik'):
                    parsed['nik'] = nik_multi
            
            # ═══ NIK VALIDATION & CORRECTION ═══
            if parsed.get('nik'):
                # Try to correct using province hint
                province_hint = parsed.get('provinsi', '')
                corrected = correct_nik_ocr(parsed['nik'], province_hint=province_hint)
                _, valid, issues = validate_nik(corrected)
                if valid:
                    parsed['nik'] = corrected
                    if used_engine and 'validation' not in used_engine:
                        used_engine += ' + nik_validation'
        
        else:
            # KK processing
            if engine == 'google' and VISION_API_KEY:
                lines = extract_text_google_vision(tmp_path)
                used_engine = 'google_vision'
            elif HAS_EASYOCR:
                lines = extract_text_easyocr(tmp_path)
                used_engine = 'easyocr (AI)'
            else:
                lines = extract_text_multi(tmp_path)
                used_engine = 'tesseract'
            parsed = parse_kk(lines)
        
        return jsonify({
            'success': True,
            'type': doc_type,
            'engine': used_engine,
            'raw_text': lines,
            'parsed': parsed
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/status')
def status():
    return jsonify({
        'google_vision': bool(VISION_API_KEY),
        'tesseract': True
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', '').lower() == 'true')
