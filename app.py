import os
import re
import json
import tempfile
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Configure tesseract
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_CMD', 'tesseract')

def preprocess_image(img_path):
    """Preprocess image for better OCR accuracy"""
    img = Image.open(img_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if too small
    width, height = img.size
    if width < 800:
        scale = 800 / width
        img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    
    # Convert to grayscale for better OCR
    img = img.convert('L')
    
    return img

def extract_text_from_image(img_path):
    """Extract text from image using Tesseract OCR"""
    img = preprocess_image(img_path)
    
    # Use Indonesian + English language
    text = pytesseract.image_to_string(img, lang='ind+eng', config='--psm 6')
    
    # Split into lines and clean
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines

def parse_ktp(texts):
    """Parse KTP fields from extracted text"""
    full_text = ' '.join(texts)
    lines = [t.strip() for t in texts if t.strip()]
    
    ktp_data = {
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
    
    for i, line in enumerate(lines):
        line_upper = line.upper().strip()
        
        # NIK - 16 digit number
        nik_match = re.search(r'\b\d{16}\b', line)
        if nik_match:
            ktp_data['nik'] = nik_match.group()
        
        # Nama
        if 'NAMA' in line_upper and ':' in line:
            nama = line.split(':', 1)[-1].strip()
            if nama and len(nama) > 2:
                ktp_data['nama'] = nama.title()
        elif 'NAMA' in line_upper and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and len(next_line) > 2 and ':' not in next_line:
                ktp_data['nama'] = next_line.title()
        
        # Tempat/Tgl Lahir
        if any(x in line_upper for x in ['TEMPAT', 'LAHIR', 'TTL']):
            if ':' in line:
                ttl = line.split(':', 1)[-1].strip()
                if ttl:
                    parts = ttl.rsplit(' ', 1)
                    if len(parts) == 2:
                        ktp_data['tempat_lahir'] = parts[0].strip().title()
                        ktp_data['tanggal_lahir'] = parts[1].strip()
                    else:
                        ktp_data['tempat_lahir'] = ttl.title()
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                parts = next_line.rsplit(' ', 1)
                if len(parts) == 2:
                    ktp_data['tempat_lahir'] = parts[0].strip().title()
                    ktp_data['tanggal_lahir'] = parts[1].strip()
        
        # Jenis Kelamin
        if 'LAKI' in line_upper or 'PEREMPUAN' in line_upper:
            if 'LAKI' in line_upper:
                ktp_data['jenis_kelamin'] = 'Laki-laki'
            else:
                ktp_data['jenis_kelamin'] = 'Perempuan'
            if ':' in line:
                jk = line.split(':', 1)[-1].strip()
                if jk:
                    ktp_data['jenis_kelamin'] = jk.title()
        
        # Golongan Darah
        if 'GOL' in line_upper and 'DARAH' in line_upper:
            if ':' in line:
                gd = line.split(':', 1)[-1].strip()
                if gd:
                    ktp_data['golongan_darah'] = gd.upper()
            elif i + 1 < len(lines):
                ktp_data['golongan_darah'] = lines[i + 1].strip().upper()
        
        # Alamat
        if 'ALAMAT' in line_upper and ':' in line:
            alamat = line.split(':', 1)[-1].strip()
            if alamat:
                ktp_data['alamat'] = alamat.title()
        
        # RT/RW
        if 'RT' in line_upper and 'RW' in line_upper:
            if ':' in line:
                rtrw = line.split(':', 1)[-1].strip()
                if rtrw:
                    ktp_data['rt_rw'] = rtrw
            elif i + 1 < len(lines):
                ktp_data['rt_rw'] = lines[i + 1].strip()
        
        # Kelurahan/Desa
        if any(x in line_upper for x in ['KELURAHAN', 'DESA']):
            if ':' in line:
                kel = line.split(':', 1)[-1].strip()
                if kel:
                    ktp_data['kelurahan'] = kel.title()
        
        # Kecamatan
        if 'KECAMATAN' in line_upper:
            if ':' in line:
                kec = line.split(':', 1)[-1].strip()
                if kec:
                    ktp_data['kecamatan'] = kec.title()
        
        # Agama
        if 'AGAMA' in line_upper:
            if ':' in line:
                agama = line.split(':', 1)[-1].strip()
                if agama:
                    ktp_data['agama'] = agama.title()
            elif i + 1 < len(lines):
                ktp_data['agama'] = lines[i + 1].strip().title()
        
        # Status Perkawinan
        if 'PERKAWINAN' in line_upper or 'KAWIN' in line_upper:
            if ':' in line:
                status = line.split(':', 1)[-1].strip()
                if status:
                    ktp_data['status_perkawinan'] = status.title()
        
        # Pekerjaan
        if 'PEKERJAAN' in line_upper:
            if ':' in line:
                kerja = line.split(':', 1)[-1].strip()
                if kerja:
                    ktp_data['pekerjaan'] = kerja.title()
            elif i + 1 < len(lines):
                ktp_data['pekerjaan'] = lines[i + 1].strip().title()
        
        # Kewarganegaraan
        if 'WARGA' in line_upper or 'KEWARGANEGARAAN' in line_upper:
            if ':' in line:
                warga = line.split(':', 1)[-1].strip()
                if warga:
                    ktp_data['kewarganegaraan'] = warga.upper()
            elif i + 1 < len(lines):
                ktp_data['kewarganegaraan'] = lines[i + 1].strip().upper()
        
        # Berlaku Hingga
        if 'BERLAKU' in line_upper:
            if ':' in line:
                berlaku = line.split(':', 1)[-1].strip()
                if berlaku:
                    ktp_data['berlaku_hingga'] = berlaku.upper()
            elif 'SEUMUR' in line_upper:
                ktp_data['berlaku_hingga'] = 'SEUMUR HIDUP'
    
    # Fallback: find NIK by pattern in full text
    if not ktp_data['nik']:
        nik_match = re.search(r'\b\d{16}\b', full_text)
        if nik_match:
            ktp_data['nik'] = nik_match.group()
    
    return ktp_data

def parse_kk(texts):
    """Parse KK fields from extracted text"""
    full_text = ' '.join(texts)
    lines = [t.strip() for t in texts if t.strip()]
    
    kk_data = {
        'nomor_kk': '',
        'nama_kepala_keluarga': '',
        'alamat': '',
        'rt_rw': '',
        'kelurahan_desa': '',
        'kecamatan': '',
        'kabupaten_kota': '',
        'provinsi': '',
        'kode_pos': '',
        'anggota_keluarga': []
    }
    
    for i, line in enumerate(lines):
        line_upper = line.upper().strip()
        
        # Nomor KK (16 digits)
        kk_match = re.search(r'\b\d{16}\b', line)
        if kk_match and not kk_data['nomor_kk']:
            kk_data['nomor_kk'] = kk_match.group()
        
        # Nama Kepala Keluarga
        if 'KEPALA' in line_upper and 'KELUARGA' in line_upper:
            if ':' in line:
                nama = line.split(':', 1)[-1].strip()
                if nama:
                    kk_data['nama_kepala_keluarga'] = nama.title()
            elif i + 1 < len(lines):
                kk_data['nama_kepala_keluarga'] = lines[i + 1].strip().title()
        
        # Alamat
        if 'ALAMAT' in line_upper and ':' in line:
            alamat = line.split(':', 1)[-1].strip()
            if alamat:
                kk_data['alamat'] = alamat.title()
        
        # RT/RW
        if 'RT' in line_upper and 'RW' in line_upper and ':' in line:
            rtrw = line.split(':', 1)[-1].strip()
            if rtrw:
                kk_data['rt_rw'] = rtrw
        
        # Kelurahan/Desa
        if any(x in line_upper for x in ['KELURAHAN', 'DESA']) and ':' in line:
            kel = line.split(':', 1)[-1].strip()
            if kel:
                kk_data['kelurahan_desa'] = kel.title()
        
        # Kecamatan
        if 'KECAMATAN' in line_upper and ':' in line:
            kec = line.split(':', 1)[-1].strip()
            if kec:
                kk_data['kecamatan'] = kec.title()
        
        # Kabupaten/Kota
        if any(x in line_upper for x in ['KABUPATEN', 'KOTA']) and ':' in line:
            kab = line.split(':', 1)[-1].strip()
            if kab:
                kk_data['kabupaten_kota'] = kab.title()
        
        # Provinsi
        if 'PROVINSI' in line_upper and ':' in line:
            prov = line.split(':', 1)[-1].strip()
            if prov:
                kk_data['provinsi'] = prov.title()
        
        # Kode Pos
        if 'KODE' in line_upper and 'POS' in line_upper:
            if ':' in line:
                kp = line.split(':', 1)[-1].strip()
                if kp:
                    kk_data['kode_pos'] = kp
    
    return kk_data

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
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Extract text
        texts = extract_text_from_image(tmp_path)
        
        # Parse based on document type
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
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
