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

# Google Vision API key (set via env var)
VISION_API_KEY = os.environ.get('GOOGLE_VISION_API_KEY', '')

def preprocess_image(img_path, mode='high_contrast'):
    """Preprocess with multiple strategies — optimized for Indonesian KTP/KK"""
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = ImageOps.exif_transpose(img)
    
    w, h = img.size
    # Downscale large images (Tesseract optimal: 1500-2500px width)
    if w > 2500:
        scale = 2000 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # Upscale small images
    elif w < 1000:
        scale = 1500 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    
    gray = img.convert('L')
    
    if mode == 'high_contrast':
        # Gentler contrast boost — preserve light text
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Sharpness(gray)
        gray = enhancer.enhance(1.5)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        # Adaptive threshold with lower cutoff — preserves thin strokes
        gray = gray.point(lambda x: 255 if x > 100 else 0, '1')
    elif mode == 'medium':
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Sharpness(gray)
        gray = enhancer.enhance(1.3)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
    elif mode == 'soft':
        # Gentle processing for well-lit photos
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.3)
        enhancer = ImageEnhance.Sharpness(gray)
        gray = enhancer.enhance(1.2)
    elif mode == 'raw':
        pass
    elif mode == 'denoise':
        # Heavy denoise for blurry/dark photos
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(2.5)
        enhancer = ImageEnhance.Brightness(gray)
        gray = enhancer.enhance(1.3)
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

def parse_ktp(lines):
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
    
    # === NIK: find 16 consecutive digits ===
    # Try multiple strategies with OCR error correction
    # Strategy 1: Direct 16 digits in full text
    nik_match = re.search(r'\b(\d{16})\b', full_text.replace(' ', ''))
    if nik_match:
        ktp['nik'] = nik_match.group(1)
    else:
        # Strategy 2: Find NIK line and fix OCR misreads
        for line in lines:
            if 'NIK' in line.upper():
                raw = line
                for ch, replacement in [('O','0'),('o','0'),('Q','0'),('l','1'),('I','1'),('|','1'),('S','5'),('s','5'),('B','8'),('G','6'),('Z','2'),('z','2')]:
                    raw = raw.replace(ch, replacement)
                digits = re.sub(r'\D', '', raw)
                if len(digits) >= 16:
                    ktp['nik'] = digits[:16]
                    break
        # Strategy 3: Look for 16-digit sequence with OCR fixes
        if not ktp['nik']:
            for line in lines:
                raw = line
                for ch, replacement in [('O','0'),('o','0'),('Q','0'),('l','1'),('I','1'),('|','1'),('S','5'),('s','5')]:
                    raw = raw.replace(ch, replacement)
                compact = raw.replace(' ', '')
                match = re.search(r'\d{16}', compact)
                if match:
                    ktp['nik'] = match.group()
                    break
        # Strategy 4: Concat all digits and find valid NIK
        if not ktp['nik']:
            all_digits = ''
            for line in lines:
                raw = line
                for ch, replacement in [('O','0'),('o','0'),('l','1'),('I','1'),('|','1')]:
                    raw = raw.replace(ch, replacement)
                all_digits += re.sub(r'\D', '', raw)
            if len(all_digits) >= 16:
                for i in range(len(all_digits) - 15):
                    candidate = all_digits[i:i+16]
                    if 11 <= int(candidate[:2]) <= 99:
                        ktp['nik'] = candidate
                        break
    
    # === NIK Fallback: Re-run OCR with different preprocessing just for NIK ===
    if not ktp['nik'] or len(re.sub(r'\D', '', ktp['nik'])) != 16:
        # Try dedicated NIK extraction with multiple preprocessing modes
        try:
            nik_candidates = []
            for mode in ['medium', 'soft', 'high_contrast', 'raw', 'denoise']:
                img = preprocess_image(img_path, mode)
                config = '--psm 6 --oem 3 -l ind+eng'
                text = pytesseract.image_to_string(img, config=config)
                # Fix common OCR misreads
                for ch, rep in [('O','0'),('o','0'),('Q','0'),('l','1'),('I','1'),('|','1'),('S','5'),('s','5'),('B','8'),('G','6'),('Z','2'),('z','2'),('D','0')]:
                    text = text.replace(ch, rep)
                # Find all digit sequences
                all_digits = re.sub(r'\D', '', text)
                # Try to find 16-digit NIK
                for i in range(len(all_digits) - 15):
                    candidate = all_digits[i:i+16]
                    if 11 <= int(candidate[:2]) <= 99:
                        nik_candidates.append(candidate)
                # Also try 17-digit sequences (OCR sometimes adds extra digit)
                for i in range(len(all_digits) - 16):
                    candidate17 = all_digits[i:i+17]
                    if 11 <= int(candidate17[:2]) <= 99:
                        # Try removing each digit to find best 16-digit match
                        for remove_pos in [8, 9, 10, 11, 12]:  # Middle positions are most likely to have extra digits
                            candidate16 = candidate17[:remove_pos] + candidate17[remove_pos+1:]
                            if 11 <= int(candidate16[:2]) <= 99:
                                nik_candidates.append(candidate16)
            # Pick most common candidate (voting)
            if nik_candidates:
                from collections import Counter
                most_common = Counter(nik_candidates).most_common(1)[0][0]
                ktp['nik'] = most_common
        except Exception:
            pass
    
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
                # Clean: only alpha + spaces
                val = re.sub(r'[^a-zA-Z\s]', '', val).strip()
                if len(val) > 2:
                    ktp['nama'] = ' '.join(w.capitalize() for w in val.split())
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Check next line isn't another field label
                if not any(kw in next_line.upper() for kw in ['NIK', 'ALAMAT', 'LAHIR', 'AGAMA', 'KELAMIN']):
                    val = re.sub(r'[^a-zA-Z\s]', '', next_line).strip()
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
                    # Fix OCR year errors
                    yr = int(year)
                    if yr < 1920 or yr > 2030:
                        yr_str = str(yr)
                        # Try fixing: 199B → 1998, 200O → 2000
                        yr_fixed = yr_str
                        for ch, rep in [('O','0'),('o','0'),('l','1'),('I','1'),('S','5'),('B','8')]:
                            yr_fixed = yr_fixed.replace(ch, rep)
                        try:
                            if 1920 <= int(yr_fixed) <= 2030:
                                ktp['tanggal_lahir'] = f"{day}-{month}-{yr_fixed}"
                        except: pass
                    # Extract tempat: everything before the date
                    tempat = val[:date_match.start()].strip().rstrip(',').strip()
                    if tempat and len(tempat) > 1:
                        ktp['tempat_lahir'] = re.sub(r'[^a-zA-Z\s.]', '', tempat).strip().upper().title()
                else:
                    # Try "BEKASI, 29-08-1998" or "BEKASI,29 AGUSTUS 1998"
                    parts = val.split(',', 1)
                    if len(parts) == 2:
                        tempat = parts[0].strip()
                        tanggal = parts[1].strip()
                        if tempat and len(tempat) > 1:
                            ktp['tempat_lahir'] = re.sub(r'[^a-zA-Z\s.]', '', tempat).strip().upper().title()
                        # Try to extract date from the second part
                        dm = re.search(r'(\d{1,2})[\s\-/.]+(\d{1,2})[\s\-/.]+(\d{4})', tanggal)
                        if dm:
                            ktp['tanggal_lahir'] = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
                        else:
                            ktp['tanggal_lahir'] = tanggal.strip()
                    elif len(val) > 2:
                        ktp['tempat_lahir'] = re.sub(r'[^a-zA-Z\s.]', '', val).strip().upper().title()
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
    for line in lines:
        upper = line.upper()
        if 'ALAMAT' in upper or 'ALAMA' in upper:
            val = find_value(line, ['Alamat', 'ALAMAT', 'Alama'])
            if val:
                # Fix OCR: "JSULTAN" → "JL. SULTAN"
                val = re.sub(r'^JL?\s*\.?\s*L?\s*([A-Z])', r'JL. \1', val)
                val = re.sub(r'^j([a-z])', r'Jl. \1', val)
                # Fix "1.JL" → "JL."
                val = re.sub(r'^\d+\.?\s*JL', 'JL.', val)
                # Clean trailing noise
                val = re.sub(r'\s+\d+\s+Te$', '', val)
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
    for line in lines:
        upper = line.upper()
        if any(kw in upper for kw in ['KELURAHAN', 'DESA', 'KEL/DESA', 'KELDESA']):
            val = find_value(line, ['Kel/Desa', 'Kelurahan', 'Desa', 'KelDesa', 'Kel'])
            if val:
                val = re.sub(r'[^a-zA-Z\s]', '', val).strip()
                if len(val) > 2:
                    ktp['kelurahan'] = val.title()
            break
    
    # === Kecamatan ===
    for line in lines:
        upper = line.upper()
        if 'KECAMATAN' in upper or 'KECAMALAN' in upper:
            val = find_value(line, ['Kecamatan', 'Kecamalan'])
            if val:
                val = re.sub(r'[^a-zA-Z\s]', '', val).strip()
                if len(val) > 2:
                    ktp['kecamatan'] = val.title()
            break
    
    # === Agama ===
    for line in lines:
        upper = line.upper()
        if 'AGAMA' in upper and 'KAWIN' not in upper:
            val = find_value(line, ['Agama'])
            if val:
                agama_upper = val.upper().strip()
                # Fix OCR: "TISLAM" → "ISLAM"
                agama_upper = agama_upper.strip().rstrip('-').strip()
                agama_map = {
                    'ISLAM': 'Islam', 'KRISTEN': 'Kristen', 'KATOLIK': 'Katolik',
                    'HINDU': 'Hindu', 'BUDHA': 'Buddha', 'BUDDHA': 'Buddha',
                    'KONGHUCU': 'Konghucu'
                }
                for key, agama_val in agama_map.items():
                    if key in agama_upper:
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
    for line in lines:
        upper = line.upper()
        if 'PEKERJAAN' in upper or 'PEKERJ' in upper:
            val = find_value(line, ['Pekerjaan', 'PEKERJAAN'])
            if val:
                # Remove trailing noise
                val = re.split(r'\s+KOTA\s+|\s+KABUPATEN\s+|\s*<.*$', val)[0]
                # Fix OCR artifacts
                val = val.replace('MM', '/M').replace('MMA', '/MA')
                val = val.replace('PC', 'PE').replace('PCLA', 'PELA')
                val = re.sub(r'\s+', ' ', val).strip()
                # Fix merged job names
                val = val.upper().replace('PELAJARMAHA', 'PELAJAR/MAHA').replace('PELAJARAMAHA', 'PELAJAR/MAHA')
                if len(val) > 2:
                    ktp['pekerjaan'] = val.title()
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
        # Choose OCR engine
        if engine == 'google' and VISION_API_KEY:
            lines = extract_text_google_vision(tmp_path)
            used_engine = 'google_vision'
        elif engine == 'google' and not VISION_API_KEY:
            # Fallback to tesseract if no API key
            lines = extract_text_multi(tmp_path)
            used_engine = 'tesseract (no Google API key)'
        else:
            lines = extract_text_multi(tmp_path)
            used_engine = 'tesseract'
        
        if doc_type == 'kk':
            parsed = parse_kk(lines)
        else:
            parsed = parse_ktp(lines)
        
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
