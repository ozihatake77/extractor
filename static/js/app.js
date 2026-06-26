// DOM
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
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
const btnTesseract = document.getElementById('btn-tesseract');
const btnGoogle = document.getElementById('btn-google');
const engineSelector = document.getElementById('engine-selector');

let currentFile = null;
let currentType = 'ktp';
let currentEngine = 'tesseract';
let lastResult = null;
let googleVisionAvailable = false;

// Check Google Vision availability on load
fetch('/api/status')
    .then(r => r.json())
    .then(data => {
        googleVisionAvailable = data.google_vision;
        if (googleVisionAvailable) {
            engineSelector.style.display = 'flex';
        }
    })
    .catch(() => {});

// Doc type
btnKtp.addEventListener('click', () => setDocType('ktp'));
btnKk.addEventListener('click', () => setDocType('kk'));
function setDocType(type) {
    currentType = type;
    btnKtp.classList.toggle('active', type === 'ktp');
    btnKk.classList.toggle('active', type === 'kk');
}

// Engine
btnTesseract.addEventListener('click', () => setEngine('tesseract'));
btnGoogle.addEventListener('click', () => setEngine('google'));
function setEngine(engine) {
    currentEngine = engine;
    btnTesseract.classList.toggle('active', engine === 'tesseract');
    btnGoogle.classList.toggle('active', engine === 'google');
}

// Upload
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
});
fileInput.addEventListener('change', (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); });

function handleFile(file) {
    if (!file.type.startsWith('image/')) { showToast('File harus gambar!', true); return; }
    if (file.size > 16 * 1024 * 1024) { showToast('Max 16MB!', true); return; }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
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
    
    loadingHint.textContent = currentEngine === 'google' 
        ? 'Google Vision sedang bekerja...' 
        : 'Tesseract sedang bekerja...';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('type', currentType);
    formData.append('engine', currentEngine);

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
    agama: 'Agama', status_perkawinan: 'Status', pekerjaan: 'Pekerjaan',
    kewarganegaraan: 'Warga Negara', berlaku_hingga: 'Berlaku Hingga',
    provinsi: 'Provinsi', kabupaten: 'Kabupaten',
    nomor_kk: 'Nomor KK', nama_kepala_keluarga: 'Kepala Keluarga',
    kelurahan_desa: 'Kelurahan', kabupaten_kota: 'Kab/Kota', kode_pos: 'Kode Pos',
    anggota_keluarga: 'Anggota Keluarga'
};

function displayResult(data) {
    loading.style.display = 'none';
    resultArea.style.display = 'block';
    resultTitle.textContent = data.type === 'kk' ? '📋 Hasil Scan KK' : '📋 Hasil Scan KTP';
    
    // Engine badge
    const engineName = data.engine || 'tesseract';
    const engineIcon = engineName.includes('google') ? '🎯' : '⚡';
    engineBadge.innerHTML = `<span class="badge">${engineIcon} ${engineName}</span>`;

    // Build table rows
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
    const text = JSON.stringify(lastResult.parsed, null, 2);
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
    fileInput.value = '';
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

// Service Worker
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/sw.js')
            .then(reg => console.log('SW registered'))
            .catch(err => console.log('SW failed:', err));
    });
}
