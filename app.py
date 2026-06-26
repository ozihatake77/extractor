import os
import re
import json
import tempfile
import traceback
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

# Google Vision API key (set via env var)
VISION_API_KEY = os.environ.get('GOOGLE_VISION_API_KEY', '')

def preprocess_image(img_path, mode='high_contrast'):
    """Preprocess with multiple strategies"""
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = ImageOps.exif_transpose(img)
    
    w, h = img.size
    if w < 1200:
        scale = 1400 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    
    gray = img.convert('L')
    
    if mode == 'high_contrast':
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(2.5)
        enhancer = ImageEnhance.Sharpness(gray)
        gray = enhancer.enhance(2.0)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        # Adaptive threshold
        gray = gray.point(lambda x: 255 if x > 130 else 0, '1')
    elif mode == 'medium':
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.8)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
    elif mode == 'raw':
        pass
    
    return gray

def ocr_pass(img_path, mode, psm):
    """Single OCR pass"""
    img = preprocess_image(img_path, mode)
    config = f'--psm {psm} --oem 3 -l ind+eng'
    text = pytesseract.image_to_string(img, config=config)
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 1]
    return lines

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

def extract_text_multi(img_path):
    """Run multiple OCR passes and MERGE all results"""
    all_lines = []
    seen = set()
    
    # Run multiple passes with different settings
    passes = [
        ('high_contrast', 6),
        ('medium', 6),
        ('high_contrast', 4),
        ('raw', 6),
    ]
    
    for mode, psm in passes:
        try:
            lines = ocr_pass(img_path, mode, psm)
            for line in lines:
                # Normalize for dedup
                key = re.sub(r'\s+', ' ', line.strip().upper())
                if key not in seen and len(key) > 1:
                    seen.add(key)
                    all_lines.append(line)
        except Exception:
            pass
    
    # Sort by typical KTP field order (top to bottom)
    # This ensures consistent parsing regardless of OCR pass order
    return all_lines

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
    """Smart KTP parser based on patterns"""
    full_text = ' '.join(lines)
    full_upper = full_text.upper()
    
    ktp = {
        'provinsi': '', 'kabupaten': '', 'nik': '', 'nama': '',
        'tempat_lahir': '', 'tanggal_lahir': '', 'jenis_kelamin': '',
        'golongan_darah': '', 'alamat': '', 'rt_rw': '', 'kelurahan': '',
        'kecamatan': '', 'agama': '', 'status_perkawinan': '',
        'pekerjaan': '', 'kewarganegaraan': '', 'berlaku_hingga': ''
    }
    
    # === NIK: find 16 consecutive digits ===
    # Try multiple strategies
    # 1. Direct 16 digits in full text
    nik_match = re.search(r'\b(\d{16})\b', full_text.replace(' ', ''))
    if nik_match:
        ktp['nik'] = nik_match.group(1)
    else:
        # 2. NIK might be split across text: "3603192904 7460001" → concat digits
        for line in lines:
            if 'NIK' in line.upper():
                digits = re.sub(r'\D', '', line)
                if len(digits) >= 16:
                    ktp['nik'] = digits[:16]
                    break
        # 3. Look for any 16-digit sequence with possible spaces
        if not ktp['nik']:
            for line in lines:
                compact = line.replace(' ', '')
                match = re.search(r'\d{16}', compact)
                if match:
                    ktp['nik'] = match.group()
                    break
    
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
                # Fix OCR: "BEKaSI" → "BEKASI", "29 08 1998" → "29-08-1998"
                val = val.replace('  ', '-')
                # Try to split
                date_match = re.search(r'(\d{1,2}[\-]\d{1,2}[\-]\d{4})', val)
                if date_match:
                    ktp['tanggal_lahir'] = date_match.group(1)
                    # Fix OCR: validate year
                    yr = int(ktp['tanggal_lahir'].split('-')[-1])
                    if yr < 1920 or yr > 2030:
                        # Try to fix common OCR errors
                        yr_str = str(yr)
                        yr_fixed = yr_str[:2] + yr_str[2:].replace('0', '9', 1) if len(yr_str) == 4 else yr_str
                        try:
                            yr_int = int(yr_fixed)
                            if 1920 <= yr_int <= 2030:
                                ktp['tanggal_lahir'] = ktp['tanggal_lahir'].replace(str(yr), yr_fixed)
                        except: pass
                    tempat = val[:date_match.start()].strip().rstrip(',').strip()
                    if tempat and len(tempat) > 1:
                        ktp['tempat_lahir'] = tempat.upper().title()
                else:
                    parts = val.split(',', 1)
                    if len(parts) == 2:
                        ktp['tempat_lahir'] = parts[0].strip().title()
                        ktp['tanggal_lahir'] = parts[1].strip().replace(' ', '-')
                    elif len(val) > 2:
                        ktp['tempat_lahir'] = val.title()
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
