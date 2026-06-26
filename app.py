import os
import re
import json
import tempfile
import traceback
from collections import OrderedDict
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import pytesseract

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
        # Choose OCR engine — EasyOCR preferred for accuracy
        if engine == 'google' and VISION_API_KEY:
            lines = extract_text_google_vision(tmp_path)
            used_engine = 'google_vision'
        elif HAS_EASYOCR:
            # EasyOCR is most accurate for Indonesian text
            lines = extract_text_easyocr(tmp_path)
            used_engine = 'easyocr (AI)'
            # Fallback to Tesseract if EasyOCR returns nothing
            if not lines:
                lines = extract_text_multi(tmp_path)
                used_engine = 'tesseract (fallback)'
        else:
            lines = extract_text_multi(tmp_path)
            used_engine = 'tesseract'
        
        if doc_type == 'kk':
            parsed = parse_kk(lines)
        else:
            parsed = parse_ktp(lines, img_path=tmp_path)
        
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
