# 📄 Extractor - Scan KTP & KK

PWA untuk scan foto KTP & KK dan convert ke text secara otomatis.

## Fitur

- 📸 Upload foto KTP / KK
- 🔍 OCR otomatis (Tesseract)
- 📋 Parse field otomatis (NIK, Nama, Alamat, dll)
- 📱 PWA - bisa diinstall di HP
- 🌙 Dark theme modern

## Tech Stack

- **Backend:** Python Flask
- **OCR:** Tesseract (pytesseract)
- **Frontend:** HTML, CSS, JavaScript (PWA)
- **Deploy:** Railway

## Cara Pakai

1. Buka aplikasi di browser
2. Pilih jenis dokumen (KTP / KK)
3. Upload foto
4. Klik "Scan Sekarang"
5. Hasil text muncul otomatis

## Deploy ke Railway

1. Push ke GitHub
2. Connect repo ke Railway
3. Railway akan auto-deploy menggunakan Dockerfile

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Install Tesseract
# Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-ind
# macOS: brew install tesseract tesseract-lang

# Run
python app.py
```

## API

### POST /api/extract

Upload gambar dan dapatkan hasil OCR.

**Request:**
- `file`: Gambar (jpg, png)
- `type`: `ktp` atau `kk`

**Response:**
```json
{
  "success": true,
  "type": "ktp",
  "raw_text": ["line1", "line2"],
  "parsed": {
    "nik": "1234567890123456",
    "nama": "John Doe",
    "alamat": "Jl. Contoh No. 123"
  }
}
```

## License

MIT
