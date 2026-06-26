import os
import re
import json
import tempfile
import traceback
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import pytesseract


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def preprocess_image(img_path, doc_type='ktp'):
    """Advanced preprocessing for better OCR"""
    img = Image.open(img_path)
    
    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Auto-rotate based on EXIF
    img = ImageOps.exif_transpose(img)
    
    # Resize - KTP/KK needs at least 1200px width for good OCR
    width, height = img.size
    target_w = 1600
    if width < target_w:
        scale = target_w / width
        img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    elif width > 3000:
        scale = 3000 / width
        img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    
    # Convert to grayscale
    gray = img.convert('L')
    
    # Increase contrast significantly
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    
    # Increase sharpness
    enhancer = ImageEnhance.Sharpness(gray)
    gray = enhancer.enhance(2.0)
    
    # Apply slight denoise with median filter
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    
    # Binarize (threshold)
    threshold = 140
    gray = gray.point(lambda x: 255 if x > threshold else 0, '1')
    
    return gray

def extract_text_ocr(img_path, doc_type='ktp'):
    """Extract text with multiple strategies"""
    img = preprocess_image(img_path, doc_type)
    
    results = []
    
    # Strategy 1: Single block (best for cards)
    try:
        config1 = '--psm 6 --oem 3 -l ind+eng'
        text1 = pytesseract.image_to_string(img, config=config1)
        lines1 = [l.strip() for l in text1.split('\n') if l.strip() and len(l.strip()) > 1]
        results.append(('psm6', lines1))
    except:
        pass
    
    # Strategy 2: Assume uniform block of text
    try:
        config2 = '--psm 4 --oem 3 -l ind+eng'
        text2 = pytesseract.image_to_string(img, config=config2)
        lines2 = [l.strip() for l in text2.split('\n') if l.strip() and len(l.strip()) > 1]
        results.append(('psm4', lines2))
    except:
        pass
    
    # Strategy 3: Raw line output
    try:
        config3 = '--psm 3 --oem 3 -l ind+eng'
        text3 = pytesseract.image_to_string(img, config=config3)
        lines3 = [l.strip() for l in text3.split('\n') if l.strip() and len(l.strip()) > 1]
        results.append(('psm3', lines3))
    except:
        pass
    
    # Pick the best result (most lines with valid content)
    best_lines = []
    best_score = 0
    
    for name, lines in results:
        score = 0
        for line in lines:
            upper = line.upper()
            # Score based on known KTP/KK keywords found
            keywords = ['NIK', 'NAMA', 'ALAMAT', 'LAHIR', 'AGAMA', 'PEKERJAAN', 
                        'KELURAHAN', 'KECAMATAN', 'RT', 'RW', 'KELUARGA', 'KEPALA',
                        'KABUPATEN', 'PROVINSI', 'KODE', 'PERKAWINAN', 'KAWIN',
                        'WARGA', 'BERLAKU', 'DARAH', 'GOL']
            for kw in keywords:
                if kw in upper:
                    score += 3
            # Score for having numbers (NIK, dates)
            if re.search(r'\d{4}', line):
                score += 1
            if len(line) > 3:
                score += 1
        if score > best_score:
            best_score = score
            best_lines = lines
    
    # Also try to get data with getdata for structured extraction
    try:
        data = pytesseract.image_to_data(img, config='--psm 6 --oem 3 -l ind+eng', output_type=pytesseract.Output.DICT)
        structured = []
        current_line = []
        last_line_num = -1
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue
            line_num = data['line_num'][i]
            if line_num != last_line_num and current_line:
                structured.append(' '.join(current_line))
                current_line = []
            current_line.append(text)
            last_line_num = line_num
        if current_line:
            structured.append(' '.join(current_line))
        
        if len(structured) > len(best_lines):
            best_lines = structured
    except:
        pass
    
    return best_lines

def clean_ocr_text(text):
    """Clean common OCR mistakes"""
    replacements = {
        '|': 'I',
        '0': 'O',  # only in specific contexts - handled in parser
        '{': '(',
        '}': ')',
        '  ': ' ',
    }
    text = text.strip()
    return text

def find_field(lines, keywords, next_line=False, colon_split=True):
    """Find a field value by keywords"""
    for i, line in enumerate(lines):
        upper = line.upper()
        for kw in keywords:
            if kw in upper:
                if colon_split and ':' in line:
                    val = line.split(':', 1)[-1].strip()
                    if val and len(val) > 1:
                        return val
                elif next_line and i + 1 < len(lines):
                    val = lines[i + 1].strip()
                    if val and ':' not in val[:5]:
                        return val
                # Try splitting by keyword
                idx = upper.find(kw)
                after = line[idx + len(kw):].strip()
                if after and len(after) > 1:
                    if after.startswith(':'):
                        after = after[1:].strip()
                    if after:
                        return after
    return ''

def parse_ktp(texts):
    """Enhanced KTP parser"""
    full_text = ' '.join(texts)
    lines = [t.strip() for t in texts if t.strip()]
    
    ktp = {
        'provinsi': '',
        'kabupaten': '',
        'nik': '',
        'nama': '',
        'tempat_lahir': '',
        'tanggal_lahir': '',
        'jenis_kelamin': '',
        'golongan_darah': '',
        'alamat': '',
        'rt_rw': '',
        'kelurahan': '',
        'kecamatan': '',
        'agama': '',
        'status_perkawinan': '',
        'pekerjaan': '',
        'kewarganegaraan': '',
        'berlaku_hingga': ''
    }
    
    # NIK - find 16 digit number (most critical)
    all_numbers = re.findall(r'\d{16}', full_text)
    if all_numbers:
        ktp['nik'] = all_numbers[0]
    else:
        # Try with OCR error correction (O->0, I->1)
        corrected = full_text.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
        all_numbers = re.findall(r'\d{16}', corrected)
        if all_numbers:
            ktp['nik'] = all_numbers[0]
    
    # Nama
    ktp['nama'] = find_field(lines, ['NAMA'])
    # Clean nama - remove non-alpha except spaces
    if ktp['nama']:
        ktp['nama'] = re.sub(r'[^a-zA-Z\s.]', '', ktp['nama']).strip().title()
    
    # TTL
    ttl = find_field(lines, ['TEMPAT', 'TGL LAHIR', 'TTL', 'LAHIR'])
    if ttl:
        # Try to split "JAKARTA, 15-08-1990" or "JAKARTA 15-08-1990"
        parts = re.split(r'[,\s]+', ttl, maxsplit=1)
        if len(parts) == 2:
            ktp['tempat_lahir'] = parts[0].strip().title()
            ktp['tanggal_lahir'] = parts[1].strip()
        else:
            ktp['tempat_lahir'] = ttl.title()
    
    # If TTL not found, try individual lines
    if not ktp['tempat_lahir']:
        for i, line in enumerate(lines):
            # Look for date pattern dd-mm-yyyy or dd/mm/yyyy
            date_match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', line)
            if date_match and i > 0:
                ktp['tanggal_lahir'] = date_match.group(0)
                # Previous line or same line might be tempat
                before_date = line[:date_match.start()].strip().rstrip(',').strip()
                if before_date and len(before_date) > 2:
                    ktp['tempat_lahir'] = before_date.title()
                elif i > 0:
                    prev = lines[i-1].strip()
                    if len(prev) > 2 and not any(x in prev.upper() for x in ['NAMA', 'NIK', 'ALAMAT']):
                        ktp['tempat_lahir'] = prev.title()
                break
    
    # Jenis Kelamin
    jk = find_field(lines, ['JENIS KELAMIN', 'KELAMIN', 'GENDER'])
    if jk:
        if 'LAKI' in jk.upper() or jk.upper().strip() in ['L', 'LK', 'LAKI']:
            ktp['jenis_kelamin'] = 'Laki-laki'
        elif 'PEREMPUAN' in jk.upper() or 'WANITA' in jk.upper() or jk.upper().strip() in ['P', 'PR']:
            ktp['jenis_kelamin'] = 'Perempuan'
        else:
            ktp['jenis_kelamin'] = jk.title()
    else:
        # Check in all lines
        for line in lines:
            upper = line.upper()
            if 'LAKI-LAKI' in upper or 'LAKI LAKI' in upper:
                ktp['jenis_kelamin'] = 'Laki-laki'
                break
            elif 'PEREMPUAN' in upper or 'WANITA' in upper:
                ktp['jenis_kelamin'] = 'Perempuan'
                break
    
    # Golongan Darah
    gd = find_field(lines, ['GOL', 'DARAH', 'GOLONGAN DARAH'])
    if gd:
        gd_clean = gd.upper().strip()
        if gd_clean in ['A', 'B', 'AB', 'O', 'A+', 'B+', 'AB+', 'O-', 'O+']:
            ktp['golongan_darah'] = gd_clean
        else:
            ktp['golongan_darah'] = gd_clean[:2] if len(gd_clean) <= 3 else ''
    
    # Alamat
    ktp['alamat'] = find_field(lines, ['ALAMAT'])
    if ktp['alamat']:
        ktp['alamat'] = ktp['alamat'].title()
    
    # RT/RW
    rtrw = find_field(lines, ['RT/RW', 'RT RW', 'RT :'])
    if rtrw:
        # Clean: "001/002" or "001 / 002"
        rtrw_clean = re.search(r'(\d{1,3})\s*/\s*(\d{1,3})', rtrw)
        if rtrw_clean:
            ktp['rt_rw'] = f"{rtrw_clean.group(1)}/{rtrw_clean.group(2)}"
        else:
            ktp['rt_rw'] = rtrw.strip()
    
    # Kelurahan
    ktp['kelurahan'] = find_field(lines, ['KELURAHAN', 'KEL/DESA', 'DESA'])
    if ktp['kelurahan']:
        ktp['kelurahan'] = re.sub(r'[^a-zA-Z\s./-]', '', ktp['kelurahan']).strip().title()
    
    # Kecamatan
    ktp['kecamatan'] = find_field(lines, ['KECAMATAN'])
    if ktp['kecamatan']:
        ktp['kecamatan'] = re.sub(r'[^a-zA-Z\s./-]', '', ktp['kecamatan']).strip().title()
    
    # Agama
    ktp['agama'] = find_field(lines, ['AGAMA'])
    if ktp['agama']:
        agama_clean = ktp['agama'].upper().strip()
        agama_map = {
            'ISLAM': 'Islam', 'KRISTEN': 'Kristen', 'KATOLIK': 'Katolik',
            'HINDU': 'Hindu', 'BUDHA': 'Buddha', 'BUDDHA': 'Buddha',
            'KONGHUCU': 'Konghucu'
        }
        for key, val in agama_map.items():
            if key in agama_clean:
                ktp['agama'] = val
                break
    
    # Status Perkawinan
    ktp['status_perkawinan'] = find_field(lines, ['PERKAWINAN', 'STATUS', 'KAWIN'])
    if ktp['status_perkawinan']:
        status = ktp['status_perkawinan'].upper()
        if 'BELUM' in status:
            ktp['status_perkawinan'] = 'Belum Kawin'
        elif 'KAWIN' in status and 'BELUM' not in status and 'CERAI' not in status:
            ktp['status_perkawinan'] = 'Kawin'
        elif 'CERAI HIDUP' in status:
            ktp['status_perkawinan'] = 'Cerai Hidup'
        elif 'CERAI MATI' in status:
            ktp['status_perkawinan'] = 'Cerai Mati'
    
    # Pekerjaan
    ktp['pekerjaan'] = find_field(lines, ['PEKERJAAN', 'KERJA'])
    if ktp['pekerjaan']:
        ktp['pekerjaan'] = ktp['pekerjaan'].title()
    
    # Kewarganegaraan
    ktp['kewarganegaraan'] = find_field(lines, ['KEWARGANEGARAAN', 'WARGA', 'WNI', 'WNA'])
    if ktp['kewarganegaraan']:
        if 'WNI' in ktp['kewarganegaraan'].upper() or 'INDONESIA' in ktp['kewarganegaraan'].upper():
            ktp['kewarganegaraan'] = 'WNI'
        elif 'WNA' in ktp['kewarganegaraan'].upper():
            ktp['kewarganegaraan'] = 'WNA'
    
    # Berlaku Hingga
    ktp['berlaku_hingga'] = find_field(lines, ['BERLAKU', 'HINGGA'])
    if ktp['berlaku_hingga']:
        if 'SEUMUR' in ktp['berlaku_hingga'].upper():
            ktp['berlaku_hingga'] = 'SEUMUR HIDUP'
    
    # Provinsi & Kabupaten (usually first 2 lines of KTP)
    for line in lines[:4]:
        upper = line.upper()
        if 'PROVINSI' in upper:
            ktp['provinsi'] = line.title()
        elif 'KABUPATEN' in upper or 'KOTA' in upper:
            ktp['kabupaten'] = line.title()
    
    return ktp

def parse_kk(texts):
    """Enhanced KK parser"""
    full_text = ' '.join(texts)
    lines = [t.strip() for t in texts if t.strip()]
    
    kk = {
        'nomor_kk': '',
        'nama_kepala_keluarga': '',
        'alamat': '',
        'rt_rw': '',
        'kelurahan_desa': '',
        'kecamatan': '',
        'kabupaten_kota': '',
        'provinsi': '',
        'kode_pos': '',
    }
    
    # Nomor KK - 16 digits
    all_numbers = re.findall(r'\d{16}', full_text)
    if all_numbers:
        kk['nomor_kk'] = all_numbers[0]
    else:
        corrected = full_text.replace('O', '0').replace('o', '0').replace('I', '1')
        all_numbers = re.findall(r'\d{16}', corrected)
        if all_numbers:
            kk['nomor_kk'] = all_numbers[0]
    
    # Nama Kepala Keluarga
    kk['nama_kepala_keluarga'] = find_field(lines, ['KEPALA KELUARGA', 'KEPALA', 'NAMA KEPALA'])
    if kk['nama_kepala_keluarga']:
        kk['nama_kepala_keluarga'] = re.sub(r'[^a-zA-Z\s.]', '', kk['nama_kepala_keluarga']).strip().title()
    
    # Alamat
    kk['alamat'] = find_field(lines, ['ALAMAT'])
    if kk['alamat']:
        kk['alamat'] = kk['alamat'].title()
    
    # RT/RW
    rtrw = find_field(lines, ['RT/RW', 'RT RW'])
    if rtrw:
        rtrw_clean = re.search(r'(\d{1,3})\s*/\s*(\d{1,3})', rtrw)
        if rtrw_clean:
            kk['rt_rw'] = f"{rtrw_clean.group(1)}/{rtrw_clean.group(2)}"
        else:
            kk['rt_rw'] = rtrw.strip()
    
    # Kelurahan/Desa
    kk['kelurahan_desa'] = find_field(lines, ['KELURAHAN', 'DESA', 'KEL/DESA'])
    if kk['kelurahan_desa']:
        kk['kelurahan_desa'] = re.sub(r'[^a-zA-Z\s./-]', '', kk['kelurahan_desa']).strip().title()
    
    # Kecamatan
    kk['kecamatan'] = find_field(lines, ['KECAMATAN'])
    if kk['kecamatan']:
        kk['kecamatan'] = re.sub(r'[^a-zA-Z\s./-]', '', kk['kecamatan']).strip().title()
    
    # Kabupaten/Kota
    kk['kabupaten_kota'] = find_field(lines, ['KABUPATEN', 'KOTA'])
    if kk['kabupaten_kota']:
        kk['kabupaten_kota'] = kk['kabupaten_kota'].title()
    
    # Provinsi
    kk['provinsi'] = find_field(lines, ['PROVINSI'])
    if kk['provinsi']:
        kk['provinsi'] = kk['provinsi'].title()
    
    # Kode Pos
    kk['kode_pos'] = find_field(lines, ['KODE POS'])
    if not kk['kode_pos']:
        # Look for 5 digit number
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
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Extract text with multiple strategies
        texts = extract_text_ocr(tmp_path, doc_type)
        
        # Parse
        if doc_type == 'kk':
            parsed = parse_kk(texts)
        else:
            parsed = parse_ktp(texts)
        
        return jsonify({
            'success': True,
            'type': doc_type,
            'raw_text': texts,
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
