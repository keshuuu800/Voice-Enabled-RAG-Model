const state = {
    isRecording: false,
    isProcessing: false,
    mediaRecorder: null,
    audioChunks: [],
    timerInterval: null,
    secondsRecorded: 0,
    isAmbientPlaying: false
};

// DOM Elements
const micBtn = document.getElementById('mic-btn');
const micStatus = document.getElementById('mic-status');
const recordingTimer = document.getElementById('recording-timer');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const voiceCenterpiece = document.getElementById('voice-centerpiece');

// Header & Controls
const ambientBtn = document.getElementById('ambient-btn');
const ambientIcon = document.getElementById('ambient-icon');
const ambientAudio = document.getElementById('ambient-audio');
const knowledgeBtn = document.getElementById('knowledge-btn');
const knowledgeModal = document.getElementById('knowledge-modal');
const closeModalBtn = document.getElementById('close-modal-btn');
const statusPill = document.getElementById('status-pill');
const statusText = document.getElementById('status-text');


// Response Sheet & Panels
const responseSheet = document.getElementById('response-sheet');
const loadingPanel = document.getElementById('loading-panel');
const loadingText = document.getElementById('loading-text');
const errorPanel = document.getElementById('error-panel');
const errorText = document.getElementById('error-text');
const transcriptPanel = document.getElementById('transcript-panel');
const transcriptText = document.getElementById('transcript-text');
const answerPanel = document.getElementById('answer-panel');
const answerText = document.getElementById('answer-text');

// Sources & Accordions
const sourcesContainer = document.getElementById('sources-container');
const sourcesList = document.getElementById('sources-list');
const contextAccordion = document.getElementById('context-accordion');
const contextToggleBtn = document.getElementById('context-toggle-btn');
const contextContent = document.getElementById('context-content');
const chunksCount = document.getElementById('chunks-count');
const chunksList = document.getElementById('chunks-list');

const latencyAccordion = document.getElementById('latency-accordion');
const latencyToggleBtn = document.getElementById('latency-toggle-btn');
const latencyContent = document.getElementById('latency-content');
const totalLatencyText = document.getElementById('total-latency-text');
const latencyBreakdownGrid = document.getElementById('latency-breakdown-grid');

// Upload Modal Elements
const uploadDropzone = document.getElementById('upload-dropzone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const uploadStatusBox = document.getElementById('upload-status');
const uploadProgressFill = document.getElementById('upload-progress-fill');
const uploadStatusText = document.getElementById('upload-status-text');
const uploadResultBadge = document.getElementById('upload-result');
const clearKbBtn = document.getElementById('clear-kb-btn');

const API_BASE = '/api';

// Initialize Application
async function init() {
    await checkHealth();
    setupEventListeners();
    setupAccordionToggles();
    setupUploadModalHandlers();
    ensureVideoAutoplay();
}

function ensureVideoAutoplay() {
    const video = document.getElementById('bg-video');
    if (video) {
        video.muted = true;
        video.play().catch(e => console.log('Autoplay muted video:', e));
    }
}

function setupEventListeners() {
    micBtn.addEventListener('click', toggleRecording);
    sendBtn.addEventListener('click', handleTextSubmit);
    textInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleTextSubmit();
    });

    // Ambient Audio Toggle
    ambientBtn.addEventListener('click', toggleAmbientAudio);

    // Knowledge Modal Toggle
    knowledgeBtn.addEventListener('click', () => knowledgeModal.classList.remove('hidden'));
    closeModalBtn.addEventListener('click', () => knowledgeModal.classList.add('hidden'));
    knowledgeModal.addEventListener('click', (e) => {
        if (e.target === knowledgeModal) knowledgeModal.classList.add('hidden');
    });
}

function setupAccordionToggles() {
    contextToggleBtn.addEventListener('click', () => {
        contextContent.classList.toggle('hidden');
        const icon = document.getElementById('context-toggle-icon');
        icon.textContent = contextContent.classList.contains('hidden') ? '⌄' : '⌃';
    });

    latencyToggleBtn.addEventListener('click', () => {
        latencyContent.classList.toggle('hidden');
    });
}

// Ambient Audio Handler
function toggleAmbientAudio() {
    if (state.isAmbientPlaying) {
        ambientAudio.pause();
        state.isAmbientPlaying = false;
        ambientIcon.textContent = '🔇';
        ambientBtn.classList.remove('glass-pill-accent');
    } else {
        ambientAudio.volume = 0.25;
        ambientAudio.play().then(() => {
            state.isAmbientPlaying = true;
            ambientIcon.textContent = '🔊';
            ambientBtn.classList.add('glass-pill-accent');
        }).catch(err => {
            console.log('Ambient audio playback prevented:', err);
        });
    }
}

// Upload Modal Handlers
function setupUploadModalHandlers() {
    browseBtn.addEventListener('click', () => fileInput.click());
    uploadDropzone.addEventListener('click', (e) => {
        if (e.target !== browseBtn) fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadDropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadDropzone.classList.remove('dragover');
        }, false);
    });

    uploadDropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files[0]) {
            handleFileUpload(dt.files[0]);
        }
    });

    // Clear Knowledge Base button
    if (clearKbBtn) {
        clearKbBtn.addEventListener('click', clearKnowledgeBase);
    }
}

async function handleFileUpload(file) {
    const MAX_SIZE = 10 * 1024 * 1024;
    if (file.size > MAX_SIZE) {
        showUploadResult(`File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds 10 MB limit.`, 'error');
        return;
    }

    const allowedExts = ['.pdf', '.txt', '.md', '.json', '.csv'];
    const fileName = file.name.toLowerCase();
    const isAllowed = allowedExts.some(ext => fileName.endsWith(ext));
    if (!isAllowed) {
        showUploadResult(`Unsupported format. Allowed: ${allowedExts.join(', ')}`, 'error');
        return;
    }

    uploadResultBadge.classList.add('hidden');
    uploadStatusBox.classList.remove('hidden');
    uploadProgressFill.style.width = '35%';
    uploadStatusText.textContent = `Uploading ${file.name}...`;

    try {
        const formData = new FormData();
        formData.append('file', file);

        uploadProgressFill.style.width = '70%';
        uploadStatusText.textContent = `Chunking & generating BGE-M3 embeddings...`;

        const response = await fetch(`${API_BASE}/upload-document`, {
            method: 'POST',
            body: formData
        });

        uploadProgressFill.style.width = '95%';

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Upload failed (Status ${response.status})`);
        }

        const data = await response.json();
        uploadProgressFill.style.width = '100%';

        setTimeout(() => {
            uploadStatusBox.classList.add('hidden');
            showUploadResult(`✅ ${data.message}`, 'success');
            fileInput.value = '';
            checkHealth();
        }, 300);

    } catch (err) {
        uploadStatusBox.classList.add('hidden');
        showUploadResult(`❌ Upload Failed: ${err.message}`, 'error');
        fileInput.value = '';
    }
}

function showUploadResult(message, type) {
    uploadResultBadge.className = `modal-result-msg ${type}`;
    uploadResultBadge.textContent = message;
    uploadResultBadge.classList.remove('hidden');
}

async function clearKnowledgeBase() {
    if (!confirm('This will remove ALL indexed documents. Are you sure?')) return;

    clearKbBtn.disabled = true;
    clearKbBtn.textContent = '⏳ Clearing...';
    uploadResultBadge.classList.add('hidden');

    try {
        const response = await fetch(`${API_BASE}/clear-knowledge`, { method: 'DELETE' });
        const data = await response.json();
        if (response.ok) {
            showUploadResult('✅ Knowledge base cleared. Upload new documents to begin.', 'success');
            checkHealth();
        } else {
            showUploadResult(`❌ Failed: ${data.detail || 'Unknown error'}`, 'error');
        }
    } catch (err) {
        showUploadResult(`❌ Network error: ${err.message}`, 'error');
    } finally {
        clearKbBtn.disabled = false;
        clearKbBtn.textContent = '🗑 Clear All Knowledge';
    }
}

// Health Check
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) throw new Error('Health check endpoint failed');
        const data = await response.json();

        if (data.status === 'ok') {
            statusPill.className = 'pill pill-active';
            const vsInfo = data?.services?.vector_store || data?.chunks || 'Indexed';
            statusText.textContent = `SYSTEM ACTIVE (${vsInfo})`;
        } else {
            statusPill.className = 'pill';
            statusText.textContent = 'LOCAL MODE';
        }
    } catch (e) {
        statusPill.className = 'pill';
        statusText.textContent = 'OFFLINE';
    }
}

// Voice Recording Flow
async function toggleRecording() {
    if (state.isProcessing) return;

    if (state.isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        state.mediaRecorder = new MediaRecorder(stream);
        state.audioChunks = [];

        state.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) state.audioChunks.push(e.data);
        };

        state.mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
            stream.getTracks().forEach(track => track.stop());
            await sendVoiceQuery(audioBlob);
        };

        state.mediaRecorder.start();
        state.isRecording = true;
        // Add recording ripple class to centerpiece
        if (voiceCenterpiece) voiceCenterpiece.classList.add('is-recording');
        setMicState('recording');


        state.secondsRecorded = 0;
        updateTimerDisplay();
        recordingTimer.classList.remove('hidden');
        state.timerInterval = setInterval(() => {
            state.secondsRecorded++;
            updateTimerDisplay();
            if (state.secondsRecorded >= 60) stopRecording();
        }, 1000);

    } catch (err) {
        console.error('Mic error:', err);
        displayError('Microphone permission denied or device unavailable.');
    }
}

function stopRecording() {
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
    }
    state.isRecording = false;
    clearInterval(state.timerInterval);
    recordingTimer.classList.add('hidden');
    // Remove recording ripple class from centerpiece
    if (voiceCenterpiece) voiceCenterpiece.classList.remove('is-recording');
    setMicState('processing');
}

function updateTimerDisplay() {
    const mins = Math.floor(state.secondsRecorded / 60).toString().padStart(2, '0');
    const secs = (state.secondsRecorded % 60).toString().padStart(2, '0');
    recordingTimer.textContent = `${mins}:${secs}`;
}

// API Calls
async function sendVoiceQuery(audioBlob) {
    displayLoading('Transcribing voice & searching Goa knowledge...');
    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');

        const response = await fetch(`${API_BASE}/voice-query`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Voice processing failed (HTTP ${response.status})`);
        }
        const data = await response.json();
        displayResults(data, true);
    } catch (err) {
        displayError(`Voice Query Error: ${err.message}`);
        setMicState('idle');
    }
}

async function sendTextQuery(query) {
    if (!query.trim()) return;
    displayLoading('Understanding your question & retrieving knowledge...');
    setMicState('processing');

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query.trim() })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Query processing failed (HTTP ${response.status})`);
        }
        const data = await response.json();
        displayResults(data, false);
    } catch (err) {
        displayError(`Text Query Error: ${err.message}`);
    } finally {
        setMicState('idle');
    }
}

function handleTextSubmit() {
    const query = textInput.value;
    if (query.trim()) {
        sendTextQuery(query);
        textInput.value = '';
    }
}

// UI State Management
function setMicState(newState) {
    micBtn.className = `mic-glass-button ${newState}`;
    state.isProcessing = (newState === 'processing');

    if (newState === 'idle') micStatus.textContent = 'Tap to speak';
    else if (newState === 'recording') micStatus.textContent = 'Listening...';
    else if (newState === 'processing') micStatus.textContent = 'Understanding...';
    else if (newState === 'answering') micStatus.textContent = 'Answering...';
}

function displayLoading(message) {
    responseSheet.classList.remove('hidden');
    loadingPanel.classList.remove('hidden');
    loadingText.textContent = message || 'Searching documents...';

    [errorPanel, transcriptPanel, sourcesContainer, contextAccordion, latencyAccordion].forEach(el => el.classList.add('hidden'));
    answerText.textContent = '';
    responseSheet.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayError(message) {
    loadingPanel.classList.add('hidden');
    errorPanel.classList.remove('hidden');
    errorText.textContent = message;
    setMicState('idle');
}

function displayResults(data, isVoice) {
    loadingPanel.classList.add('hidden');
    errorPanel.classList.add('hidden');
    setMicState('answering');
    setTimeout(() => setMicState('idle'), 1200);

    responseSheet.classList.remove('hidden');

    // Transcript ("YOU SAID")
    if (isVoice && data.transcript) {
        transcriptPanel.classList.remove('hidden');
        transcriptText.textContent = `"${data.transcript}"`;
    } else if (data.query) {
        transcriptPanel.classList.remove('hidden');
        transcriptText.textContent = `"${data.query}"`;
    } else {
        transcriptPanel.classList.add('hidden');
    }

    // Grounded Answer
    answerText.textContent = data.answer;

    // Source Citations
    if (data.sources && data.sources.length > 0) {
        sourcesContainer.classList.remove('hidden');
        sourcesList.innerHTML = data.sources.map(s => `
            <div class="source-pill">
                <span>📄 ${escapeHtml(s.source)} ${s.page ? '(Page ' + s.page + ')' : ''}</span>
            </div>
        `).join('');
    } else {
        sourcesContainer.classList.add('hidden');
    }

    // Retrieved Chunks Context
    if (data.chunks && data.chunks.length > 0) {
        contextAccordion.classList.remove('hidden');
        chunksCount.textContent = data.chunks.length;
        chunksList.innerHTML = data.chunks.map((c, idx) => `
            <div class="chunk-card">
                <div class="chunk-meta">Chunk ${idx + 1} • ${escapeHtml(c.source)} ${c.page ? 'Page ' + c.page : ''} • Score: ${c.score.toFixed(4)}</div>
                <div class="chunk-body">${escapeHtml(c.text)}</div>
            </div>
        `).join('');
    } else {
        contextAccordion.classList.add('hidden');
    }

    // Pipeline Latency
    if (data.latency) {
        latencyAccordion.classList.remove('hidden');
        totalLatencyText.textContent = `${data.latency.total_ms.toFixed(1)} ms`;
        renderLatencyGrid(data.latency);
    } else {
        latencyAccordion.classList.add('hidden');
    }

    responseSheet.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderLatencyGrid(latency) {
    const stages = [
        { key: 'stt_ms', label: 'Speech-to-Text (Sarvam)' },
        { key: 'embedding_ms', label: 'Query Embedding (BGE-M3)' },
        { key: 'bm25_ms', label: 'BM25 Keyword Search' },
        { key: 'vector_ms', label: 'ChromaDB Vector Search' },
        { key: 'rrf_ms', label: 'Reciprocal Rank Fusion (RRF)' },
        { key: 'llm_ms', label: 'LLM Answer Generation' }
    ];

    let html = '';
    stages.forEach(stage => {
        const val = latency[stage.key];
        if (val !== undefined && val !== null && val > 0) {
            html += `
                <div class="latency-item-row">
                    <span>${stage.label}</span>
                    <strong>${val.toFixed(1)} ms</strong>
                </div>
            `;
        }
    });

    latencyBreakdownGrid.innerHTML = html;
}

function escapeHtml(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Start
init();
