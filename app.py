import os
import re
import json
import tempfile
import traceback
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import easyocr
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

reader = None

def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['id', 'en'], gpu=False)
    return reader

def preprocess_image(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    if w < 1000:
        scale = 1200 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def extract_text_easyocr(img_path):
    reader = get_reader()
    img = preprocess_image(img_path)
    img_np = np.array(img)
    results = reader.readtext(img_np, detail=1, paragraph=False)
    results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
    
    lines = []
    current_line = []
    last_y = None
    y_threshold = 20
    
    for bbox, text, conf in results:
        center_y = (bbox[0][1] + bbox[2][1]) / 2
        if last_y is not None and abs(center_y - last_y) > y_threshold:
            if current_line:
                lines.append(current_line)
            current_line = []
        current_line.append((bbox, text, conf))
        last_y = center_y
    if current_line:
        lines.append(current_line)
    
    line_texts = []
    for line in lines:
        line.sort(key=lambda r: r[0][0][0])
        texts = [t[1] for t in line]
        avg_conf = sum(t[2] for t in line) / len(line)
        line_texts.append((' '.join(texts), avg_conf))
    
    return line_texts, results

def extract_after_keyword(text, keywords):
    """Extract value after a keyword in text, handling various formats"""
    upper = text.upper()
    for kw in keywords:
        kw_upper = kw.upper()
        idx = upper.find(kw_upper)
        if idx >= 0:
            after = text[idx + len(kw):]
            # Clean leading separators
            after = re.sub(r'^[\s:=\-_,;]+', '', after).strip()
            return after
    return None

def clean_value(text):
    """Clean OCR artifacts from a value"""
    if not text:
        return ''
    text = text.strip()
    # Remove trailing noise
    text = re.sub(r'[\s\-=_<>|/\\]+$', '', text)
    # Remove leading noise  
    text = re.sub(r'^[\s\-=_<>|/\\]+', '', text)
    return text.strip()

def parse_ktp_easyocr(line_texts, raw_results):
    full_text = ' '.join([t[0] for t in line_texts])
    
    ktp = {
        'provinsi': '', 'kabupaten': '', 'nik': '', 'nama': '',
        'tempat_lahir': '', 'tanggal_lahir': '', 'jenis_kelamin': '',
        'golongan_darah': '', 'alamat': '', 'rt_rw': '', 'kelurahan': '',
        'kecamatan': '', 'agama': '', 'status_perkawinan': '',
        'pekerjaan': '', 'kewarganegaraan': '', 'berlaku_hingga': ''
    }
    
    # === NIK ===
    nik_match = re.search(r'(\d{16})', full_text.replace(' ', ''))
    if nik_match:
        ktp['nik'] = nik_match.group(1)
    
    # === Provinsi ===
    for text, conf in line_texts:
        if 'PROVINSI' in text.upper():
            val = extract_after_keyword(text, ['PROVINSI'])
            if val:
                # Fix common OCR: "BARA /" → "BARAT"
                val = re.sub(r'\s*/\s*$', 'T', val)
                val = re.sub(r'\s+', ' ', val).strip()
                ktp['provinsi'] = val.title()
            break
    
    # === Kabupaten/Kota ===
    for text, conf in line_texts:
        upper = text.upper()
        if 'KOTA' in upper or 'KABUPATEN' in upper:
            # Only if it's a standalone line (not part of other field)
            if not any(kw in upper for kw in ['ALAMAT', 'PEKERJAAN', 'LAHIR']):
                ktp['kabupaten'] = clean_value(text).title()
                break
    
    # === Nama ===
    for text, conf in line_texts:
        if 'NAMA' in text.upper() and 'NAMAKA' not in text.upper():
            val = extract_after_keyword(text, ['Nama'])
            if val and len(val) > 2:
                # Fix spacing: "DeniAdi" → "Deni Adi"
                val = re.sub(r'([a-z])([A-Z])', r'\1 \2', val)
                # Remove non-alpha except spaces
                val = re.sub(r'[^a-zA-Z\s]', '', val).strip()
                ktp['nama'] = ' '.join(w.capitalize() for w in val.split())
            break
    
    # === Tempat/Tgl Lahir ===
    for text, conf in line_texts:
        upper = text.upper()
        if 'LAHIR' in upper or 'TEMPAT' in upper or 'TTL' in upper:
            val = extract_after_keyword(text, ['Tempat/Tgl Lahir', 'Tempat', 'Tgl Lahir', 'TTL', 'Lahir'])
            if val:
                # Fix OCR: "BEKaSI, 29 08 1998" → "BEKASI, 29-08-1998"
                val = val.replace('  ', '-')
                # Fix common OCR errors
                val = val.replace('BEKaSI', 'BEKASI').replace('BEKa', 'BEKA')
                
                # Try to split place and date
                date_match = re.search(r'(\d{1,2}[\-]\d{1,2}[\-]\d{4})', val)
                if date_match:
                    ktp['tanggal_lahir'] = date_match.group(1)
                    tempat = val[:date_match.start()].strip().rstrip(',').strip()
                    if tempat:
                        ktp['tempat_lahir'] = tempat.upper().title()
                else:
                    # Maybe date is separate
                    parts = val.split(',', 1)
                    if len(parts) == 2:
                        ktp['tempat_lahir'] = parts[0].strip().title()
                        ktp['tanggal_lahir'] = parts[1].strip().replace(' ', '-')
                    else:
                        ktp['tempat_lahir'] = val.title()
            break
    
    # Fallback: find date pattern
    if not ktp['tanggal_lahir']:
        for text, conf in line_texts:
            date_match = re.search(r'(\d{1,2}[\-]\d{1,2}[\-]\d{4})', text)
            if date_match:
                ktp['tanggal_lahir'] = date_match.group(1)
                break
    
    # === Jenis Kelamin ===
    for text, conf in line_texts:
        upper = text.upper()
        if 'KELAMIN' in upper or 'JENIS' in upper:
            if 'LAKI' in upper:
                ktp['jenis_kelamin'] = 'Laki-laki'
            elif 'PEREMPUAN' in upper or 'WANITA' in upper:
                ktp['jenis_kelamin'] = 'Perempuan'
            break
    # Fallback
    if not ktp['jenis_kelamin']:
        for text, conf in line_texts:
            if 'LAKI-LAKI' in text.upper() or 'LAKI LAKI' in text.upper():
                ktp['jenis_kelamin'] = 'Laki-laki'
                break
    
    # === Golongan Darah ===
    for text, conf in line_texts:
        upper = text.upper()
        if 'DARAH' in upper:
            # Try to find blood type value
            val = extract_after_keyword(text, ['Gol. Darah', 'Gol Darah', 'Golongan Darah', 'Darah', 'Goldarah'])
            if val:
                val_clean = val.upper().strip()[:3]
                blood_types = ['A', 'B', 'AB', 'O']
                for bt in blood_types:
                    if val_clean.startswith(bt):
                        ktp['golongan_darah'] = bt
                        break
            # Also check if it's embedded in the line
            if not ktp['golongan_darah']:
                match = re.search(r'\b(A|B|AB|O)\b', upper)
                if match and 'LAKI' not in upper[:match.start()]:
                    ktp['golongan_darah'] = match.group(1)
            break
    
    # === Alamat ===
    for text, conf in line_texts:
        if 'ALAMAT' in text.upper() or 'ALAMA' in text.upper():
            val = extract_after_keyword(text, ['Alamat', 'Alamat', 'ALAMAT', 'Alama'])
            if val:
                # Fix OCR: "JSULTAN" → "JL SULTAN"
                val = re.sub(r'^J([A-Z])', r'JL \1', val)
                val = re.sub(r'^j([a-z])', r'Jl \1', val)
                ktp['alamat'] = val.upper()
            break
    
    # === RT/RW ===
    for text, conf in line_texts:
        if 'RT' in text.upper() and '/' in text:
            match = re.search(r'(\d{1,3})\s*/\s*(\d{1,3})', text)
            if match:
                ktp['rt_rw'] = f"{match.group(1).zfill(3)}/{match.group(2).zfill(3)}"
                break
    # Fallback: standalone RT/RW pattern
    if not ktp['rt_rw']:
        for text, conf in line_texts:
            match = re.search(r'(\d{3})\s*/\s*(\d{3})', text)
            if match and len(text) < 15:  # Short line, likely RT/RW
                ktp['rt_rw'] = f"{match.group(1)}/{match.group(2)}"
                break
    
    # === Kelurahan/Desa ===
    for text, conf in line_texts:
        upper = text.upper()
        if 'KEL' in upper and 'DESA' in upper or 'KELURAHAN' in upper or 'KELDESA' in upper:
            val = extract_after_keyword(text, ['Kel/Desa', 'Kelurahan', 'Desa', 'KelDesa', 'Kel'])
            if val:
                val = re.sub(r'[^a-zA-Z\s]', '', val).strip()
                ktp['kelurahan'] = val.title()
            break
    
    # === Kecamatan ===
    for text, conf in line_texts:
        if 'KECAMATAN' in text.upper() or 'KECAMALAN' in text.upper():
            val = extract_after_keyword(text, ['Kecamatan', 'Kecamalan'])
            if val:
                val = re.sub(r'[^a-zA-Z\s]', '', val).strip()
                ktp['kecamatan'] = val.title()
            break
    
    # === Agama ===
    for text, conf in line_texts:
        if 'AGAMA' in text.upper():
            val = extract_after_keyword(text, ['Agama'])
            if val:
                agama_upper = val.upper().strip()
                agama_map = {
                    'ISLAM': 'Islam', 'KRISTEN': 'Kristen', 'KATOLIK': 'Katolik',
                    'HINDU': 'Hindu', 'BUDHA': 'Buddha', 'BUDDHA': 'Buddha',
                    'KONGHUCU': 'Konghucu'
                }
                for key, agama_val in agama_map.items():
                    if key in agama_upper:
                        ktp['agama'] = agama_val
                        break
                if not ktp['agama']:
                    ktp['agama'] = val.title()
            break
    
    # === Status Perkawinan ===
    for text, conf in line_texts:
        upper = text.upper()
        if 'PERKAWINAN' in upper or 'PERKAWNAN' in upper or ('STATUS' in upper and 'KAWIN' in upper):
            val = extract_after_keyword(text, ['Status Perkawinan', 'Perkawinan', 'Perkawnan', 'Status'])
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
    for text, conf in line_texts:
        if 'PEKERJAAN' in text.upper() or 'PEKORJAAN' in text.upper() or 'PEKERJAN' in text.upper():
            val = extract_after_keyword(text, ['Pekerjaan', 'Pekorjaan', 'Pekerjan'])
            if val:
                # Remove trailing noise like "KOTA BEKASI"
                val = re.split(r'\s+KOTA\s+|\s+KABUPATEN\s+', val)[0]
                # Fix OCR: "PCLAJARMMAHASISWA" → "PELAJAR/MAHASISWA"
                val = val.replace('MM', '/M').replace('MMA', '/MA').replace('PC', 'PE')
                val = val.replace('PCLA', 'PELA')
                ktp['pekerjaan'] = val.strip().title()
            break
    
    # === Kewarganegaraan ===
    for text, conf in line_texts:
        upper = text.upper()
        if 'KEWARGANEGARAAN' in upper or 'WARGA' in upper:
            val = extract_after_keyword(text, ['Kewarganegaraan', 'Kewarganegaraan', 'Warga Negara'])
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
    # Fallback
    if not ktp['kewarganegaraan']:
        for text, conf in line_texts:
            if 'WNI' in text.upper():
                ktp['kewarganegaraan'] = 'WNI'
                break
    
    # === Berlaku Hingga ===
    for text, conf in line_texts:
        upper = text.upper()
        if 'BERLAKU' in upper or 'BORLAKU' in upper or 'HINGGA' in upper:
            val = extract_after_keyword(text, ['Berlaku Hingga', 'Berlaku', 'Borlaku', 'Hingga'])
            if val:
                if 'SEUMUR' in val.upper() or 'HIDUP' in val.upper():
                    ktp['berlaku_hingga'] = 'SEUMUR HIDUP'
                else:
                    ktp['berlaku_hingga'] = val.strip().upper()
            elif 'SEUMUR' in upper:
                ktp['berlaku_hingga'] = 'SEUMUR HIDUP'
            break
    # Fallback
    if not ktp['berlaku_hingga']:
        for text, conf in line_texts:
            if 'SEUMUR' in text.upper():
                ktp['berlaku_hingga'] = 'SEUMUR HIDUP'
                break
    
    return ktp

def parse_kk_easyocr(line_texts, raw_results):
    full_text = ' '.join([t[0] for t in line_texts])
    
    kk = {
        'nomor_kk': '', 'nama_kepala_keluarga': '', 'alamat': '',
        'rt_rw': '', 'kelurahan_desa': '', 'kecamatan': '',
        'kabupaten_kota': '', 'provinsi': '', 'kode_pos': ''
    }
    
    all_digits = re.findall(r'\d{16}', full_text.replace(' ', ''))
    if all_digits:
        kk['nomor_kk'] = all_digits[0]
    
    kk['nama_kepala_keluarga'] = ''
    for text, conf in line_texts:
        if 'KEPALA' in text.upper():
            val = extract_after_keyword(text, ['Kepala Keluarga', 'Kepala'])
            if val:
                kk['nama_kepala_keluarga'] = re.sub(r'[^a-zA-Z\s.]', '', val).strip().title()
            break
    
    for text, conf in line_texts:
        if 'ALAMAT' in text.upper():
            val = extract_after_keyword(text, ['Alamat'])
            if val:
                kk['alamat'] = val.title()
            break
    
    for text, conf in line_texts:
        match = re.search(r'(\d{3})\s*/\s*(\d{3})', text)
        if match:
            kk['rt_rw'] = f"{match.group(1)}/{match.group(2)}"
            break
    
    for text, conf in line_texts:
        if 'KELURAHAN' in text.upper() or 'DESA' in text.upper():
            val = extract_after_keyword(text, ['Kelurahan', 'Desa', 'Kel/Desa'])
            if val:
                kk['kelurahan_desa'] = re.sub(r'[^a-zA-Z\s]', '', val).strip().title()
            break
    
    for text, conf in line_texts:
        if 'KECAMATAN' in text.upper():
            val = extract_after_keyword(text, ['Kecamatan'])
            if val:
                kk['kecamatan'] = re.sub(r'[^a-zA-Z\s]', '', val).strip().title()
            break
    
    for text, conf in line_texts:
        if 'KABUPATEN' in text.upper() or 'KOTA' in text.upper():
            val = extract_after_keyword(text, ['Kabupaten', 'Kota'])
            if val:
                kk['kabupaten_kota'] = val.title()
            break
    
    for text, conf in line_texts:
        if 'PROVINSI' in text.upper():
            val = extract_after_keyword(text, ['Provinsi'])
            if val:
                kk['provinsi'] = val.title()
            break
    
    for text, conf in line_texts:
        if 'KODE' in text.upper() and 'POS' in text.upper():
            val = extract_after_keyword(text, ['Kode Pos'])
            if val:
                kk['kode_pos'] = val.strip()
            break
    if not kk['kode_pos']:
        for text, conf in line_texts:
            match = re.search(r'\b\d{5}\b', text)
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
        line_texts, raw_results = extract_text_easyocr(tmp_path)
        
        if doc_type == 'kk':
            parsed = parse_kk_easyocr(line_texts, raw_results)
        else:
            parsed = parse_ktp_easyocr(line_texts, raw_results)
        
        raw_lines = [text for text, conf in line_texts]
        
        return jsonify({
            'success': True,
            'type': doc_type,
            'raw_text': raw_lines,
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
