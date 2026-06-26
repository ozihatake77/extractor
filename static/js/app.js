// DOM
const uploadArea = document.getElementById('upload-area');
const fileCamera = document.getElementById('file-camera');
const fileGallery = document.getElementById('file-gallery');
const btnCamera = document.getElementById('btn-camera');
const btnGallery = document.getElementById('btn-gallery');
const previewArea = document.getElementById('preview-area');
const previewImg = document.getElementById('preview-img');
const btnScan = document.getElementById('btn-scan');
const btnReset = document.getElementById('btn-reset');
const loading = document.getElementById('loading');
const loadingHint = document.getElementById('loading-hint');
const resultArea = document.getElementById('result-area');
const resultTitle = document.getElementById('result-title');
const engineBadge = document.getElementById('engine-badge');
const parsedData = document.getElementById('parsed-data');
const rawText = document.getElementById('raw-text');
const btnCopy = document.getElementById('btn-copy');
const btnNew = document.getElementById('btn-new');
const btnKtp = document.getElementById('btn-ktp');
const btnKk = document.getElementById('btn-kk');
const scannedCard = document.getElementById('scanned-card');
const scannedImg = document.getElementById('scanned-img');

let currentFile = null;
let currentType = 'ktp';
let lastResult = null;
let previewDataUrl = null;

// Doc type
btnKtp.addEventListener('click', () => setDocType('ktp'));
btnKk.addEventListener('click', () => setDocType('kk'));
function setDocType(type) {
    currentType = type;
    btnKtp.classList.toggle('active', type === 'ktp');
    btnKk.classList.toggle('active', type === 'kk');
}

// Camera button
btnCamera.addEventListener('click', (e) => {
    e.preventDefault();
    fileCamera.value = '';
    fileCamera.style.display = 'block';
    fileCamera.click();
    setTimeout(() => fileCamera.style.display = 'none', 1000);
});
fileCamera.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});

// Gallery button
btnGallery.addEventListener('click', (e) => {
    e.preventDefault();
    fileGallery.value = '';
    fileGallery.style.display = 'block';
    fileGallery.click();
    setTimeout(() => fileGallery.style.display = 'none', 1000);
});
fileGallery.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});

// Drag & drop on upload area
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) { showToast('File harus gambar!', true); return; }
    if (file.size > 16 * 1024 * 1024) { showToast('Max 16MB!', true); return; }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewDataUrl = e.target.result;
        previewImg.src = previewDataUrl;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';
        resultArea.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Scan
btnScan.addEventListener('click', async () => {
    if (!currentFile) return;
    previewArea.style.display = 'none';
    loading.style.display = 'block';
    resultArea.style.display = 'none';

    loadingHint.textContent = 'Tesseract sedang bekerja...';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('type', currentType);
    formData.append('engine', 'tesseract');

    try {
        const response = await fetch('/api/extract', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        lastResult = data;
        displayResult(data);
    } catch (error) {
        showToast('Error: ' + error.message, true);
        resetToUpload();
    }
});

// Display result
const fieldLabels = {
    nik: 'NIK', nama: 'Nama', tempat_lahir: 'Tempat Lahir', tanggal_lahir: 'Tanggal Lahir',
    jenis_kelamin: 'Jenis Kelamin', golongan_darah: 'Gol. Darah', alamat: 'Alamat',
    rt_rw: 'RT/RW', kelurahan: 'Kelurahan', kecamatan: 'Kecamatan',
    kabupaten: 'Kabupaten', provinsi: 'Provinsi',
    agama: 'Agama', status_perkawinan: 'Status', pekerjaan: 'Pekerjaan',
    kewarganegaraan: 'Warga Negara', berlaku_hingga: 'Berlaku Hingga',
    nomor_kk: 'Nomor KK', nama_kepala_keluarga: 'Kepala Keluarga',
    kelurahan_desa: 'Kelurahan', kabupaten_kota: 'Kab/Kota', kode_pos: 'Kode Pos',
    anggota_keluarga: 'Anggota Keluarga'
};

function displayResult(data) {
    loading.style.display = 'none';
    resultArea.style.display = 'block';
    resultTitle.textContent = data.type === 'kk' ? '📋 Hasil Scan KK' : '📋 Hasil Scan KTP';

    // Show scanned image
    if (previewDataUrl) {
        scannedImg.src = previewDataUrl;
        scannedCard.style.display = 'block';
    } else {
        scannedCard.style.display = 'none';
    }

    // Engine badge
    const engineName = data.engine || 'tesseract';
    const engineIcon = engineName.includes('google') ? '🎯' : '⚡';
    engineBadge.innerHTML = `<span class="badge">${engineIcon} ${engineName}</span>`;

    // Build table rows (order comes from Python OrderedDict)
    let html = '';
    for (const [key, value] of Object.entries(data.parsed)) {
        if (key === 'anggota_keluarga') continue;
        const label = fieldLabels[key] || key;
        const display = (value && value !== '-') ? value : '-';
        const cls = (value && value !== '-') ? 'has-value' : 'empty';
        html += `<tr><td>${label}</td><td class="${cls}">${display}</td></tr>`;
    }
    parsedData.innerHTML = html;

    // Raw text
    rawText.textContent = (data.raw_text || []).join('\n');
}

// Copy
btnCopy.addEventListener('click', async () => {
    if (!lastResult) return;
    const lines = [];
    for (const [key, value] of Object.entries(lastResult.parsed)) {
        if (key === 'anggota_keluarga') continue;
        const label = fieldLabels[key] || key;
        if (value && value !== '-') lines.push(`${label}: ${value}`);
    }
    const text = lines.join('\n');
    try {
        await navigator.clipboard.writeText(text);
        showToast('Berhasil dicopy!');
    } catch {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        showToast('Berhasil dicopy!');
    }
});

// Reset
btnReset.addEventListener('click', resetToUpload);
btnNew.addEventListener('click', resetToUpload);
function resetToUpload() {
    currentFile = null;
    previewDataUrl = null;
    fileCamera.value = '';
    fileGallery.value = '';
    uploadArea.style.display = 'block';
    previewArea.style.display = 'none';
    loading.style.display = 'none';
    resultArea.style.display = 'none';
}

// Toast
function showToast(message, isError = false) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = 'toast' + (isError ? ' error' : '');
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => { toast.classList.remove('show'); setTimeout(() => toast.remove(), 300); }, 3000);
}

// ═══ NFC Reader (Web NFC API — Chrome Android) ═══
const btnNfc = document.getElementById('btn-nfc');
let nfcReader = null;

// Show NFC button if Web NFC is supported
if ('NDEFReader' in window) {
    btnNfc.style.display = 'flex';
    btnNfc.addEventListener('click', startNfcScan);
}

async function startNfcScan() {
    if (!('NDEFReader' in window)) {
        showToast('NFC tidak didukung di browser ini. Gunakan Chrome di Android.', true);
        return;
    }
    
    try {
        btnNfc.classList.add('nfc-scanning');
        btnNfc.textContent = '📡 Scanning...';
        showToast('Dekatkan KTP ke belakang HP...');
        
        const ndef = new NDEFReader();
        
        // Set timeout
        const timeout = setTimeout(() => {
            btnNfc.classList.remove('nfc-scanning');
            btnNfc.textContent = '📡 NFC';
            showToast('Timeout — coba lagi', true);
        }, 30000);
        
        await ndef.scan();
        
        ndef.addEventListener('reading', ({ serialNumber, message }) => {
            clearTimeout(timeout);
            btnNfc.classList.remove('nfc-scanning');
            btnNfc.textContent = '📡 NFC';
            
            let nfcData = {
                uid: serialNumber,
                records: []
            };
            
            // Parse NDEF records
            if (message && message.records) {
                for (const record of message.records) {
                    if (record.recordType === 'text') {
                        const textDecoder = new TextDecoder(record.encoding || 'utf-8');
                        nfcData.records.push(textDecoder.decode(record.data));
                    } else if (record.recordType === 'url') {
                        const textDecoder = new TextDecoder();
                        nfcData.records.push(textDecoder.decode(record.data));
                    }
                }
            }
            
            // Show NFC result
            displayNfcResult(nfcData);
        });
        
    } catch (error) {
        btnNfc.classList.remove('nfc-scanning');
        btnNfc.textContent = '📡 NFC';
        if (error.name === 'NotAllowedError') {
            showToast('Izinkan akses NFC di pengaturan browser', true);
        } else if (error.name === 'NotSupportedError') {
            showToast('NFC tidak didukung di device ini', true);
        } else {
            showToast('Error NFC: ' + error.message, true);
        }
    }
}

function displayNfcResult(nfcData) {
    loading.style.display = 'none';
    resultArea.style.display = 'block';
    resultTitle.textContent = '📋 Hasil NFC Scan';
    scannedCard.style.display = 'none';
    engineBadge.innerHTML = '<span class="badge">📡 NFC Reader</span>';
    
    let html = '';
    html += `<tr><td>UID</td><td class="has-value">${nfcData.uid || '-'}</td></tr>`;
    
    if (nfcData.records.length > 0) {
        html += `<tr><td>Data</td><td class="has-value">${nfcData.records.join('<br>')}</td></tr>`;
    } else {
        html += `<tr><td>Data</td><td class="empty">Chip terproteksi — butuh foto juga untuk data lengkap</td></tr>`;
    }
    
    parsedData.innerHTML = html;
    rawText.textContent = `UID: ${nfcData.uid}\nRecords: ${JSON.stringify(nfcData.records)}`;
    
    showToast('NFC berhasil dibaca! UID: ' + nfcData.uid);
}

// Service Worker
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/sw.js')
            .then(reg => console.log('SW registered'))
            .catch(err => console.log('SW failed:', err));
    });
}
