// DOM Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewArea = document.getElementById('preview-area');
const previewImg = document.getElementById('preview-img');
const btnScan = document.getElementById('btn-scan');
const btnReset = document.getElementById('btn-reset');
const loading = document.getElementById('loading');
const resultArea = document.getElementById('result-area');
const parsedData = document.getElementById('parsed-data');
const rawText = document.getElementById('raw-text');
const btnCopy = document.getElementById('btn-copy');
const btnNew = document.getElementById('btn-new');
const btnKtp = document.getElementById('btn-ktp');
const btnKk = document.getElementById('btn-kk');

let currentFile = null;
let currentType = 'ktp';
let lastResult = null;

// Document type selector
btnKtp.addEventListener('click', () => setDocType('ktp'));
btnKk.addEventListener('click', () => setDocType('kk'));

function setDocType(type) {
    currentType = type;
    btnKtp.classList.toggle('active', type === 'ktp');
    btnKk.classList.toggle('active', type === 'kk');
}

// Upload area click
uploadArea.addEventListener('click', () => fileInput.click());

// Drag & drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Handle file selection
function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('File harus berupa gambar!', 'error');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
        showToast('Ukuran file maksimal 16MB!', 'error');
        return;
    }
    
    currentFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';
        resultArea.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Scan button
btnScan.addEventListener('click', async () => {
    if (!currentFile) return;
    
    // Show loading
    previewArea.style.display = 'none';
    loading.style.display = 'block';
    resultArea.style.display = 'none';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('type', currentType);
    
    try {
        const response = await fetch('/api/extract', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        lastResult = data;
        displayResult(data);
        
    } catch (error) {
        showToast('Error: ' + error.message, 'error');
        resetToUpload();
    }
});

// Display result
function displayResult(data) {
    loading.style.display = 'none';
    resultArea.style.display = 'block';
    
    // Display parsed data
    const parsed = data.parsed;
    let tableHTML = '';
    
    const fieldLabels = {
        'provinsi': 'Provinsi',
        'kabupaten': 'Kabupaten',
        'nik': 'NIK',
        'nama': 'Nama',
        'tempat_lahir': 'Tempat Lahir',
        'tanggal_lahir': 'Tanggal Lahir',
        'jenis_kelamin': 'Jenis Kelamin',
        'golongan_darah': 'Golongan Darah',
        'alamat': 'Alamat',
        'rt_rw': 'RT/RW',
        'kelurahan': 'Kelurahan',
        'kecamatan': 'Kecamatan',
        'agama': 'Agama',
        'status_perkawinan': 'Status Perkawinan',
        'pekerjaan': 'Pekerjaan',
        'kewarganegaraan': 'Kewarganegaraan',
        'berlaku_hingga': 'Berlaku Hingga',
        'nomor_kk': 'Nomor KK',
        'nama_kepala_keluarga': 'Nama Kepala Keluarga',
        'kelurahan_desa': 'Kelurahan/Desa',
        'kabupaten_kota': 'Kabupaten/Kota',
        'kode_pos': 'Kode Pos'
    };
    
    for (const [key, value] of Object.entries(parsed)) {
        if (key === 'anggota_keluarga') continue;
        const label = fieldLabels[key] || key;
        const displayValue = value || '-';
        const emptyClass = value ? '' : ' empty';
        tableHTML += `
            <div class="data-row">
                <span class="data-label">${label}</span>
                <span class="data-value${emptyClass}">${displayValue}</span>
            </div>
        `;
    }
    
    parsedData.innerHTML = tableHTML;
    
    // Display raw text
    rawText.textContent = data.raw_text.join('\n');
}

// Copy button
btnCopy.addEventListener('click', async () => {
    if (!lastResult) return;
    
    try {
        await navigator.clipboard.writeText(JSON.stringify(lastResult.parsed, null, 2));
        showToast('JSON berhasil dicopy!');
    } catch (err) {
        // Fallback
        const textArea = document.createElement('textarea');
        textArea.value = JSON.stringify(lastResult.parsed, null, 2);
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        showToast('JSON berhasil dicopy!');
    }
});

// Reset buttons
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

// Toast notification
function showToast(message, type = 'success') {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();
    
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    if (type === 'error') {
        toast.style.background = 'var(--danger)';
    }
    document.body.appendChild(toast);
    
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Register Service Worker
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/sw.js')
            .then(reg => console.log('Service Worker registered'))
            .catch(err => console.log('SW registration failed:', err));
    });
}
