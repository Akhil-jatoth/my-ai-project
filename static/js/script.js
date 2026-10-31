/* =====================================================
   DOM Elements
   ===================================================== */

const imageInput = document.getElementById('imageInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const downloadBtn = document.getElementById('downloadBtn');
const downloadHeatmapBtn = document.getElementById('downloadHeatmapBtn');
const copyReportBtn = document.getElementById('copyReportBtn');
const printReportBtn = document.getElementById('printReportBtn');
const clearFileBtn = document.getElementById('clearFileBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');
const fileInfo = document.getElementById('fileInfo');
const placeholderContent = document.getElementById('placeholderContent');
const resultsContainer = document.getElementById('resultsContainer');
const statusDot = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const fileDropzone = document.getElementById('fileDropzone');
const totalAnalysesCount = document.getElementById('totalAnalysesCount');
const lastProcessingTime = document.getElementById('lastProcessingTime');
const systemDevice = document.getElementById('systemDevice');
const modeBadge = document.getElementById('modeBadge');

const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const crowdCount = document.getElementById('crowdCount');
const densityLevel = document.getElementById('densityLevel');
const alertStatus = document.getElementById('alertStatus');
const alertBanner = document.getElementById('alertBanner');
const alertMessage = document.getElementById('alertMessage');
const alertCard = document.querySelector('.alert-card');
const resultTimestamp = document.getElementById('resultTimestamp');
const processingTime = document.getElementById('processingTime');

const originalImage = document.getElementById('originalImage');
const heatmapImage = document.getElementById('heatmapImage');
const sliderOriginal = document.getElementById('sliderOriginal');
const sliderHeatmap = document.getElementById('sliderHeatmap');

const tabUpload = document.getElementById('tabUpload');
const uploadPanel = document.getElementById('uploadPanel');
const webcamPanel = document.getElementById('webcamPanel');
const webcamVideo = document.getElementById('webcamVideo');
const webcamCanvas = document.getElementById('webcamCanvas');
const webcamPlaceholder = document.getElementById('webcamPlaceholder');
const startWebcamBtn = document.getElementById('startWebcamBtn');
const captureBtn = document.getElementById('captureBtn');

const compareContainer = document.getElementById('compareContainer');
const compareOverlay = document.getElementById('compareOverlay');
const compareHandle = document.getElementById('compareHandle');
const sliderView = document.getElementById('sliderView');
const sideView = document.getElementById('sideView');
const viewSlider = document.getElementById('viewSlider');
const viewSide = document.getElementById('viewSide');

const fullscreenModal = document.getElementById('fullscreenModal');
const modalImage = document.getElementById('modalImage');
const modalClose = document.getElementById('modalClose');
const modalBackdrop = document.getElementById('modalBackdrop');
const fullscreenOriginal = document.getElementById('fullscreenOriginal');
const fullscreenHeatmap = document.getElementById('fullscreenHeatmap');

const shortcutsModal = document.getElementById('shortcutsModal');
const shortcutsBtn = document.getElementById('shortcutsBtn');
const shortcutsClose = document.getElementById('shortcutsClose');
const shortcutsBackdrop = document.getElementById('shortcutsBackdrop');

const heatmapOpacity = document.getElementById('heatmapOpacity');
const opacityValue = document.getElementById('opacityValue');
const printReport = document.getElementById('printReport');

const recommendationsList = document.getElementById('recommendationsList');
const recCount = document.getElementById('recCount');
const historyList = document.getElementById('historyList');
const historyCount = document.getElementById('historyCount');
const toast = document.getElementById('toast');

const modelStatusPill = document.getElementById('modelStatusPill');
const systemEngine = document.getElementById('systemEngine');
const systemDeviceDetail = document.getElementById('systemDeviceDetail');
const systemModel = document.getElementById('systemModel');

/* =====================================================
   Global State
   ===================================================== */

let selectedFile = null;
let currentPrediction = null;
let webcamStream = null;
let historyChart = null;
let distributionChart = null;
let pipelineInterval = null;
let sliderPosition = 50;
const GAUGE_CIRCUMFERENCE = 326.73;

/* =====================================================
   Input Tabs
   ===================================================== */

document.querySelectorAll('.input-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.input-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const isUpload = tab.dataset.tab === 'upload';
        uploadPanel.classList.toggle('hidden', !isUpload);
        webcamPanel.classList.toggle('hidden', isUpload);

        if (isUpload) stopWebcam();
    });
});

/* =====================================================
   File Input
   ===================================================== */

imageInput.addEventListener('change', handleFileSelect);
clearFileBtn.addEventListener('click', () => {
    imageInput.value = '';
    clearFileInfo();
});

fileDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileDropzone.classList.add('drag-over');
});

fileDropzone.addEventListener('dragleave', () => {
    fileDropzone.classList.remove('drag-over');
});

fileDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    fileDropzone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        handleFileSelect();
    }
});

function handleFileSelect() {
    const file = imageInput.files[0];
    if (!file) { clearFileInfo(); return; }

    if (!isValidImageFile(file)) {
        showError('Invalid file type. Please select PNG, JPG, JPEG, GIF, or BMP.');
        clearFileInfo();
        return;
    }
    if (file.size > 50 * 1024 * 1024) {
        showError('File too large. Maximum size is 50 MB.');
        clearFileInfo();
        return;
    }

    selectedFile = file;
    displayFileInfo(file);
    analyzeBtn.disabled = false;
    hideError();
}

function isValidImageFile(file) {
    return ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'].includes(file.type);
}

function displayFileInfo(file) {
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.classList.remove('hidden');
}

function clearFileInfo() {
    fileInfo.classList.add('hidden');
    analyzeBtn.disabled = true;
    selectedFile = null;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

/* =====================================================
   Webcam
   ===================================================== */

startWebcamBtn.addEventListener('click', toggleWebcam);
captureBtn.addEventListener('click', captureFromWebcam);

async function toggleWebcam() {
    if (webcamStream) {
        stopWebcam();
        return;
    }
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
            audio: false
        });
        webcamVideo.srcObject = webcamStream;
        webcamPlaceholder.classList.add('hidden');
        startWebcamBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Camera';
        captureBtn.disabled = false;
    } catch {
        showError('Camera access denied or unavailable.');
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }
    webcamVideo.srcObject = null;
    webcamPlaceholder.classList.remove('hidden');
    startWebcamBtn.innerHTML = '<i class="fas fa-play"></i> Start Camera';
    captureBtn.disabled = true;
}

function captureFromWebcam() {
    if (!webcamStream) return;
    const ctx = webcamCanvas.getContext('2d');
    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    ctx.drawImage(webcamVideo, 0, 0);

    webcamCanvas.toBlob(blob => {
        if (!blob) return;
        selectedFile = new File([blob], `webcam-capture-${Date.now()}.png`, { type: 'image/png' });
        displayFileInfo(selectedFile);
        analyzeBtn.disabled = false;
        hideError();
        showToast('Frame captured successfully');
    }, 'image/png');
}

/* =====================================================
   Pipeline Loading
   ===================================================== */

function startPipelineAnimation() {
    const steps = document.querySelectorAll('.pipeline-step');
    let current = 0;
    steps.forEach(s => s.classList.remove('active', 'done'));

    pipelineInterval = setInterval(() => {
        if (current > 0) steps[current - 1].classList.replace('active', 'done');
        if (current < steps.length) {
            steps[current].classList.add('active');
            current++;
        }
    }, 600);
}

function stopPipelineAnimation() {
    clearInterval(pipelineInterval);
    document.querySelectorAll('.pipeline-step').forEach(s => {
        s.classList.remove('active');
        s.classList.add('done');
    });
}

/* =====================================================
   Analysis
   ===================================================== */

analyzeBtn.addEventListener('click', analyzeImage);

async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select or capture an image first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    const sceneCapacityInput = document.getElementById('sceneCapacityInput');
    if (sceneCapacityInput && sceneCapacityInput.value.trim()) {
        formData.append('scene_capacity', sceneCapacityInput.value.trim());
    }

    showLoading(true);
    hideError();
    startPipelineAnimation();

    try {
        const response = await fetch('/api/predict', { method: 'POST', body: formData });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP Error ${response.status}`);
        }

        const result = await response.json();
        if (result.success) {
            currentPrediction = result;
            displayResults(result);
            await loadHistory();
        } else {
            showError(result.error || 'Prediction failed.');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message.startsWith('Error:') ? error.message : `Error: ${error.message}`);
    } finally {
        stopPipelineAnimation();
        showLoading(false);
    }
}

/* =====================================================
   Display Results
   ===================================================== */

function getLevelClass(level) {
    const map = { 'Low': 'level-low', 'Medium': 'level-medium', 'High': 'level-high', 'Very High': 'level-very-high' };
    return map[level] || 'level-medium';
}

function getRiskClass(level) {
    const map = { 'Moderate': 'risk-moderate', 'High': 'risk-high', 'Critical': 'risk-critical' };
    return map[level] || 'risk-moderate';
}

function getRecIcon(type) {
    const map = { critical: 'fa-triangle-exclamation', warning: 'fa-exclamation-circle', success: 'fa-circle-check', info: 'fa-circle-info' };
    return map[type] || 'fa-circle-info';
}

function formatTimestamp(iso) {
    if (!iso) return '';
    return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function animateCounter(el, target, duration = 800) {
    const startTime = performance.now();
    function update(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(target * eased).toLocaleString();
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function updateGauge(percent) {
    const fill = document.getElementById('gaugeFill');
    const value = document.getElementById('occupancyValue');
    const barFill = document.getElementById('capacityBarFill');
    const barLabel = document.getElementById('capacityBarLabel');

    const offset = GAUGE_CIRCUMFERENCE - (percent / 100) * GAUGE_CIRCUMFERENCE;
    const color = percent > 75 ? 'var(--danger)' : percent > 50 ? 'var(--warning)' : 'var(--success)';

    fill.style.strokeDashoffset = offset;
    fill.style.stroke = color;
    barFill.style.width = percent + '%';
    barFill.style.background = color;
    barLabel.textContent = percent + '% of estimated max capacity';

    let current = 0;
    const startTime = performance.now();
    function tick(now) {
        const progress = Math.min((now - startTime) / 800, 1);
        current = Math.round(percent * (1 - Math.pow(1 - progress, 3)));
        value.textContent = current + '%';
        if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

function displayResults(prediction) {
    animateCounter(crowdCount, prediction.count);

    // Uncertainty display (Phase 8)
    const crowdUncertainty = document.getElementById('crowdUncertainty');
    if (crowdUncertainty && prediction.uncertainty) {
        crowdUncertainty.textContent = `Uncertainty: ±${prediction.uncertainty.delta} (${prediction.uncertainty.lower_bound} – ${prediction.uncertainty.upper_bound})`;
    }

    // Show DEMO badge next to the timestamp when in demo mode
    const isDemo = prediction.mode === 'demo';
    let existingBadge = document.getElementById('demoResultBadge');
    if (isDemo) {
        if (!existingBadge) {
            existingBadge = document.createElement('span');
            existingBadge.id = 'demoResultBadge';
            existingBadge.className = 'demo-result-badge';
            existingBadge.innerHTML = '<i class="fas fa-flask"></i> DEMO';
            const meta = document.querySelector('.result-meta');
            if (meta) meta.appendChild(existingBadge);
        }
        existingBadge.classList.remove('hidden');
    } else if (existingBadge) {
        existingBadge.classList.add('hidden');
    }

    densityLevel.textContent = prediction.level;
    densityLevel.className = 'kpi-value kpi-badge ' + getLevelClass(prediction.level);

    if (prediction.alert) {
        alertStatus.textContent = 'Overcrowded';
        alertCard.classList.add('danger');
        alertBanner.classList.remove('hidden', 'alert-success');
        alertBanner.classList.add('alert-warning');
        alertMessage.textContent = prediction.alert_message;
    } else {
        alertStatus.textContent = 'Normal';
        alertCard.classList.remove('danger');
        alertBanner.classList.remove('hidden', 'alert-warning');
        alertBanner.classList.add('alert-success');
        alertMessage.textContent = prediction.alert_message;
    }

    originalImage.src = prediction.original_image;
    heatmapImage.src = prediction.heatmap_image;
    sliderOriginal.src = prediction.original_image;
    sliderHeatmap.src = prediction.heatmap_image;
    sliderHeatmap.onload = syncCompareImages;
    sliderOriginal.onload = syncCompareImages;
    initCompareSlider();
    applyHeatmapOpacity();

    if (prediction.timestamp) resultTimestamp.textContent = formatTimestamp(prediction.timestamp);
    
    // Timing breakdown (Phase 18)
    if (prediction.timing_breakdown && prediction.timing_breakdown.total_ms) {
        const t = prediction.timing_breakdown;
        const totalSec = (t.total_ms / 1000).toFixed(2);
        processingTime.textContent = `${totalSec}s (infer: ${t.inference_ms.toFixed(0)}ms)`;
        lastProcessingTime.textContent = `${totalSec}s`;
    } else if (prediction.processing_time) {
        processingTime.textContent = prediction.processing_time + 's';
        lastProcessingTime.textContent = prediction.processing_time + 's';
    }

    // Capacity & Occupancy (Phase 10)
    if (prediction.scene_capacity) {
        if (prediction.scene_capacity.configured && prediction.scene_capacity.occupancy_percentage !== null) {
            updateGauge(prediction.scene_capacity.occupancy_percentage);
            const barLabel = document.getElementById('capacityBarLabel');
            if (barLabel) barLabel.textContent = `${prediction.scene_capacity.occupancy_percentage}% of configured venue capacity (${prediction.scene_capacity.capacity.toLocaleString()})`;
        } else {
            const fill = document.getElementById('gaugeFill');
            const value = document.getElementById('occupancyValue');
            const barFill = document.getElementById('capacityBarFill');
            const barLabel = document.getElementById('capacityBarLabel');
            if (fill) fill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
            if (value) value.textContent = 'N/A';
            if (barFill) barFill.style.width = '0%';
            if (barLabel) barLabel.textContent = 'Capacity not configured — Occupancy unavailable';
        }
    } else if (prediction.occupancy_percent !== undefined) {
        updateGauge(prediction.occupancy_percent);
    }

    // Reliability Score (Phase 7)
    const relScore = prediction.reliability_score ?? prediction.confidence;
    if (relScore !== undefined) {
        document.getElementById('confidenceValue').textContent = relScore + '%';
        document.getElementById('confidenceDetail').textContent = relScore + '%';
        const fill = document.getElementById('confidenceFill');
        fill.style.width = relScore + '%';
        fill.style.background = relScore > 80 ? 'var(--success)' : relScore > 60 ? 'var(--warning)' : 'var(--danger)';
    }

    if (prediction.risk_zones) {
        const rz = prediction.risk_zones;
        const riskBadge = document.getElementById('riskBadge');
        riskBadge.textContent = rz.risk_level;
        riskBadge.className = 'risk-badge ' + getRiskClass(rz.risk_level);
        document.getElementById('riskAreaPercentage').textContent = rz.risk_area_percentage + '%';
        document.getElementById('maxDensity').textContent = Math.round(rz.max_density).toLocaleString();
        document.getElementById('avgDensity').textContent = Math.round(rz.avg_density).toLocaleString();
        document.getElementById('riskZonesCount').textContent = (rz.risk_zones_count || 0).toLocaleString();
    }

    // Spatial Risk Zones List (Phase 11)
    const zonesList = document.getElementById('highDensityZonesList');
    if (zonesList) {
        const zones = prediction.spatial_risk_zones || [];
        if (zones.length > 0) {
            zonesList.innerHTML = zones.map(z => `
                <div class="zone-badge-item ${z.severity.toLowerCase()}">
                    <div>
                        <strong>Zone ${z.zone}</strong> (${z.severity})
                        <span style="font-size: 0.72rem; color: var(--text-muted); display: block;">Grid BBox: [${z.bbox.join(', ')}]</span>
                    </div>
                    <span style="font-weight: 600;">${z.area_percentage}% of scene</span>
                </div>
            `).join('');
        } else {
            zonesList.innerHTML = '<p class="diag-empty">No high-density clusters detected in the spatial grid.</p>';
        }
    }

    // Image Quality Diagnostics (Phase 13)
    if (prediction.image_quality) {
        const q = prediction.image_quality;
        const badge = document.getElementById('imageQualityBadge');
        if (badge) {
            badge.textContent = q.is_low_quality ? 'Quality: Degraded' : 'Quality: Normal';
            badge.className = 'system-pill ' + (q.is_low_quality ? 'demo' : 'live');
        }
        const res = document.getElementById('imgResolution');
        if (res) res.textContent = `${q.width} × ${q.height}`;
        const sharp = document.getElementById('imgSharpness');
        if (sharp) sharp.textContent = Math.round(q.sharpness).toLocaleString();
        const bright = document.getElementById('imgBrightness');
        if (bright) bright.textContent = Math.round(q.brightness).toString();
        const contrast = document.getElementById('imgContrast');
        if (contrast) contrast.textContent = Math.round(q.contrast).toString();

        const advList = document.getElementById('qualityAdvisoriesList');
        if (advList) {
            if (q.advisories && q.advisories.length > 0) {
                advList.innerHTML = q.advisories.map(a => `<div class="quality-advisory-item"><i class="fas fa-circle-exclamation"></i> ${a}</div>`).join('');
            } else {
                advList.innerHTML = '';
            }
        }
    }

    if (prediction.metrics) {
        const m = prediction.metrics;
        document.getElementById('densityPerSqm').textContent = m.density_per_sqm + ' /m²';
        document.getElementById('crowdSpread').textContent = m.crowd_spread_score.toFixed(1) + ' / 100';
        document.getElementById('growthPotential').textContent = m.growth_potential.toFixed(1) + '%';
        document.getElementById('maxCapacity').textContent = m.max_capacity_estimate.toLocaleString();
        document.getElementById('availableSpace').textContent = (m.available_space ?? 0).toFixed(1) + '%';
    }

    if (prediction.density_distribution) {
        updateDistributionChart(prediction.density_distribution);
    }

    if (prediction.recommendations) {
        recCount.textContent = prediction.recommendations.length + ' item' + (prediction.recommendations.length !== 1 ? 's' : '');
        recommendationsList.innerHTML = prediction.recommendations.map(rec => `
            <div class="recommendation-item ${rec.type}">
                <div class="recommendation-icon"><i class="fas ${getRecIcon(rec.type)}"></i></div>
                <div class="recommendation-body">
                    <strong>${rec.title}</strong>
                    <p>${rec.text}</p>
                </div>
            </div>
        `).join('');
    }

    placeholderContent.classList.add('hidden');
    resultsContainer.classList.remove('hidden');
    setTimeout(() => resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

/* =====================================================
   Heatmap Opacity
   ===================================================== */

heatmapOpacity.addEventListener('input', () => {
    opacityValue.textContent = heatmapOpacity.value + '%';
    applyHeatmapOpacity();
});

function applyHeatmapOpacity() {
    const opacity = heatmapOpacity.value / 100;
    if (sliderHeatmap) sliderHeatmap.style.opacity = opacity;
    if (heatmapImage) heatmapImage.style.opacity = opacity;
}

/* =====================================================
   Compare Slider
   ===================================================== */

let sliderDragging = false;

function syncCompareImages() {
    const w = compareContainer.offsetWidth;
    if (!w) return;
    sliderHeatmap.style.width = w + 'px';
}

function initCompareSlider() {
    setSliderPosition(50);
    syncCompareImages();
}

window.addEventListener('resize', syncCompareImages);

function setSliderPosition(percent) {
    sliderPosition = Math.max(5, Math.min(95, percent));
    compareOverlay.style.width = sliderPosition + '%';
    compareHandle.style.left = sliderPosition + '%';
}

compareContainer.addEventListener('mousedown', () => { sliderDragging = true; });
document.addEventListener('mouseup', () => { sliderDragging = false; });
document.addEventListener('mousemove', (e) => {
    if (!sliderDragging) return;
    const rect = compareContainer.getBoundingClientRect();
    setSliderPosition(((e.clientX - rect.left) / rect.width) * 100);
});

compareContainer.addEventListener('touchstart', () => { sliderDragging = true; });
document.addEventListener('touchend', () => { sliderDragging = false; });
document.addEventListener('touchmove', (e) => {
    if (!sliderDragging) return;
    const rect = compareContainer.getBoundingClientRect();
    setSliderPosition(((e.touches[0].clientX - rect.left) / rect.width) * 100);
});

viewSlider.addEventListener('click', () => {
    viewSlider.classList.add('active');
    viewSide.classList.remove('active');
    sliderView.classList.remove('hidden');
    sideView.classList.add('hidden');
});

viewSide.addEventListener('click', () => {
    viewSide.classList.add('active');
    viewSlider.classList.remove('active');
    sideView.classList.remove('hidden');
    sliderView.classList.add('hidden');
});

/* =====================================================
   Charts
   ===================================================== */

function updateDistributionChart(dist) {
    document.getElementById('distLow').textContent = dist.low + '%';
    document.getElementById('distMed').textContent = dist.medium + '%';
    document.getElementById('distHigh').textContent = dist.high + '%';

    const ctx = document.getElementById('distributionChart');
    if (distributionChart) distributionChart.destroy();

    distributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low', 'Medium', 'High'],
            datasets: [{
                data: [dist.low, dist.medium, dist.high],
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                borderWidth: 0,
                spacing: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: '68%',
            plugins: { legend: { display: false } }
        }
    });
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        if (!response.ok) return;
        const data = await response.json();
        renderHistory(data);
    } catch { /* silent */ }
}

const clearHistoryBtn = document.getElementById('clearHistoryBtn');
if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', async () => {
        if (!confirm('Clear all analysis history records?')) return;
        try {
            const res = await fetch('/api/history/clear', { method: 'POST' });
            if (res.ok) {
                showToast('History cleared');
                await loadHistory();
            }
        } catch (e) {
            console.error('Failed to clear history:', e);
        }
    });
}

function renderHistory(data) {
    const history = data.history || [];
    totalAnalysesCount.textContent = data.total_analyses || 0;
    historyCount.textContent = history.length + ' run' + (history.length !== 1 ? 's' : '');

    if (history.length === 0) {
        historyList.innerHTML = '<p class="history-empty">No analyses yet this session</p>';
        if (historyChart) { historyChart.destroy(); historyChart = null; }
        return;
    }

    historyList.innerHTML = [...history].reverse().map(h => `
        <div class="history-item">
            <span class="history-item-name" title="${h.filename}">${h.filename}</span>
            <span class="history-item-level ${h.level.toLowerCase().replace(' ', '-')}">${h.level}</span>
            <span class="history-item-conf">${h.confidence || '—'}%</span>
            <span class="history-item-count">${h.count.toLocaleString()}</span>
        </div>
    `).join('');

    const ctx = document.getElementById('historyChart');
    if (historyChart) historyChart.destroy();

    historyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: history.map((_, i) => '#' + (i + 1)),
            datasets: [{
                label: 'Crowd Count',
                data: history.map(h => h.count),
                backgroundColor: history.map(h =>
                    h.level === 'Low' ? '#10b981' :
                    h.level === 'Medium' ? '#f59e0b' : '#ef4444'
                ),
                borderRadius: 6,
                barThickness: 14
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 10, family: 'DM Sans' } } },
                y: { beginAtZero: true, grid: { color: '#f1f5f9' }, ticks: { font: { size: 10, family: 'DM Sans' } } }
            }
        }
    });
}

/* =====================================================
   Actions
   ===================================================== */

resetBtn.addEventListener('click', resetForm);

function resetForm() {
    imageInput.value = '';
    selectedFile = null;
    currentPrediction = null;
    clearFileInfo();
    resultsContainer.classList.add('hidden');
    placeholderContent.classList.remove('hidden');
    hideError();
    fileDropzone.classList.remove('drag-over');
    alertBanner.classList.add('hidden');
    stopWebcam();
}

downloadBtn.addEventListener('click', downloadResults);

function downloadResults() {
    if (!currentPrediction) { showError('No results to export.'); return; }

    const results = {
        timestamp: currentPrediction.timestamp || new Date().toISOString(),
        crowd_count: currentPrediction.count,
        density_level: currentPrediction.level,
        alert_active: currentPrediction.alert,
        alert_message: currentPrediction.alert_message,
        confidence: currentPrediction.confidence,
        occupancy_percent: currentPrediction.occupancy_percent,
        risk_zones: currentPrediction.risk_zones,
        metrics: currentPrediction.metrics,
        recommendations: currentPrediction.recommendations,
        density_distribution: currentPrediction.density_distribution,
        processing_time: currentPrediction.processing_time
    };

    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' }));
    link.download = `crowd-analysis-${Date.now()}.json`;
    link.click();
    showToast('JSON report downloaded');
}

downloadHeatmapBtn.addEventListener('click', () => {
    if (!currentPrediction?.heatmap_image) { showError('No heatmap to save.'); return; }
    const link = document.createElement('a');
    link.href = currentPrediction.heatmap_image;
    link.download = `heatmap-${Date.now()}.png`;
    link.click();
    showToast('Heatmap saved');
});

function buildReportText(p) {
    return [
        '=== CrowdSense Analysis Report ===',
        `Time: ${formatTimestamp(p.timestamp)}`,
        `Crowd Count: ${p.count.toLocaleString()}`,
        `Density Level: ${p.level}`,
        `Safety Status: ${p.alert ? 'OVERCROWDED' : 'Normal'}`,
        `Occupancy: ${p.occupancy_percent}%`,
        `AI Confidence: ${p.confidence}%`,
        `Risk Level: ${p.risk_zones?.risk_level || 'N/A'}`,
        `High-Risk Area: ${p.risk_zones?.risk_area_percentage || 0}%`,
        `Density/m²: ${p.metrics?.density_per_sqm || 'N/A'}`,
        `Available Space: ${p.metrics?.available_space ?? 'N/A'}%`,
        `Processing Time: ${p.processing_time}s`,
        '',
        'Recommendations:',
        ...(p.recommendations || []).map(r => `• ${r.title}: ${r.text}`)
    ].join('\n');
}

copyReportBtn.addEventListener('click', async () => {
    if (!currentPrediction) { showError('No results to copy.'); return; }
    try {
        await navigator.clipboard.writeText(buildReportText(currentPrediction));
        showToast('Report copied to clipboard');
    } catch {
        showError('Failed to copy report.');
    }
});

printReportBtn.addEventListener('click', () => {
    if (!currentPrediction) { showError('No results to print.'); return; }
    const p = currentPrediction;

    printReport.innerHTML = `
        <div style="font-family: 'DM Sans', sans-serif; color: #0f172a;">
            <h1 style="font-size: 1.5rem; margin-bottom: 0.25rem;">CrowdSense Analysis Report</h1>
            <p style="color: #64748b; margin-bottom: 1.5rem;">${formatTimestamp(p.timestamp)} · Processed in ${p.processing_time}s</p>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 1.5rem;">
                <tr><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0; font-weight: 600;">Crowd Count</td><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;">${p.count.toLocaleString()}</td></tr>
                <tr><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0; font-weight: 600;">Density Level</td><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;">${p.level}</td></tr>
                <tr><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0; font-weight: 600;">Safety Status</td><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;">${p.alert ? 'OVERCROWDED' : 'Normal'}</td></tr>
                <tr><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0; font-weight: 600;">Occupancy</td><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;">${p.occupancy_percent}%</td></tr>
                <tr><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0; font-weight: 600;">AI Confidence</td><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;">${p.confidence}%</td></tr>
                <tr><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0; font-weight: 600;">Risk Level</td><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;">${p.risk_zones?.risk_level || 'N/A'}</td></tr>
                <tr><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0; font-weight: 600;">Available Space</td><td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;">${p.metrics?.available_space ?? 'N/A'}%</td></tr>
            </table>
            <h2 style="font-size: 1rem; margin-bottom: 0.75rem;">Recommendations</h2>
            <ul style="padding-left: 1.25rem; color: #475569;">
                ${(p.recommendations || []).map(r => `<li style="margin-bottom: 0.5rem;"><strong>${r.title}</strong> — ${r.text}</li>`).join('')}
            </ul>
        </div>
    `;

    window.print();
    showToast('Print dialog opened');
});

/* =====================================================
   Fullscreen & Shortcuts Modals
   ===================================================== */

function openFullscreen(src) {
    modalImage.src = src;
    fullscreenModal.classList.remove('hidden');
}

fullscreenOriginal.addEventListener('click', () => openFullscreen(originalImage.src));
fullscreenHeatmap.addEventListener('click', () => openFullscreen(heatmapImage.src));
modalClose.addEventListener('click', () => fullscreenModal.classList.add('hidden'));
modalBackdrop.addEventListener('click', () => fullscreenModal.classList.add('hidden'));

shortcutsBtn.addEventListener('click', () => shortcutsModal.classList.remove('hidden'));
shortcutsClose.addEventListener('click', () => shortcutsModal.classList.add('hidden'));
shortcutsBackdrop.addEventListener('click', () => shortcutsModal.classList.add('hidden'));

/* =====================================================
   Utilities
   ===================================================== */

function showLoading(show) {
    loadingSpinner.classList.toggle('hidden', !show);
    analyzeBtn.disabled = show || !selectedFile;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
    errorMessage.textContent = '';
}

function showToast(message) {
    toast.textContent = message;
    toast.classList.remove('hidden');
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.classList.add('hidden'), 300);
    }, 2500);
}

/* =====================================================
   Health Check & System Info
   ===================================================== */

async function checkServerHealth() {
    try {
        const response = await fetch('/api/health');
        if (response.ok) {
            const data = await response.json();
            // Support both new lowercase mode values ('demo'/'trained') and legacy uppercase
            const isDemo = (data.mode === 'demo' || data.mode === 'DEMO');

            statusDot.classList.remove('offline');
            statusDot.classList.add('pulse');
            statusText.textContent = isDemo ? 'Demo Mode' : 'System Online';

            modeBadge.classList.remove('hidden', 'production');
            modeBadge.textContent = isDemo ? 'Demo' : 'Live';
            if (!isDemo) modeBadge.classList.add('production');

            const deviceLabel = (data.device || '').includes('cuda') ? 'GPU' : 'CPU';
            systemDevice.textContent = deviceLabel;
            systemDeviceDetail.textContent = data.device || 'CPU';
            systemEngine.textContent = isDemo ? 'Demo Engine' : 'Neural Network';

            // Show trained vs demo status clearly (Section 17)
            if (data.validation_status) {
                systemModel.textContent = data.validation_status;
            } else {
                const weightsLoaded = data.weights_loaded || data.model_loaded;
                if (weightsLoaded) {
                    systemModel.textContent = 'MCNN • Trained Model';
                } else {
                    systemModel.textContent = 'MCNN • Weights unavailable';
                }
            }

            modelStatusPill.textContent = isDemo ? 'Demo' : 'Live';
            modelStatusPill.className = 'system-pill ' + (isDemo ? 'demo' : 'live');

            // Show or hide the demo notice banner
            const demoNoticeBanner = document.getElementById('demoNoticeBanner');
            if (demoNoticeBanner) {
                demoNoticeBanner.classList.toggle('hidden', !isDemo);
            }
        } else {
            setOfflineStatus();
        }
    } catch {
        setOfflineStatus();
    }
}

function setOfflineStatus() {
    statusDot.classList.add('offline');
    statusDot.classList.remove('pulse');
    statusText.textContent = 'Offline';
    modelStatusPill.textContent = 'Offline';
    modelStatusPill.className = 'system-pill demo';
}

/* =====================================================
   Model Performance (Phase 5)
   ===================================================== */

async function loadModelPerformance() {
    const perfEmptyMsg = document.getElementById('perfEmptyMsg');
    const perfMetricsGrid = document.getElementById('perfMetricsGrid');
    const perfStatusPill = document.getElementById('perfStatusPill');

    try {
        const res = await fetch('/api/model-performance');
        if (res.ok) {
            const data = await res.json();
            if (data.status === 'evaluated' && data.metrics) {
                if (perfStatusPill) {
                    perfStatusPill.textContent = 'Evaluated';
                    perfStatusPill.className = 'system-pill live';
                }
                if (perfEmptyMsg) perfEmptyMsg.classList.add('hidden');
                if (perfMetricsGrid) {
                    perfMetricsGrid.classList.remove('hidden');
                    document.getElementById('perfDataset').textContent = data.metrics.dataset || 'Benchmark';
                    document.getElementById('perfSamples').textContent = data.metrics.evaluated_samples || '—';
                    document.getElementById('perfMAE').textContent = data.metrics.mae != null ? data.metrics.mae.toFixed(2) : '—';
                    document.getElementById('perfRMSE').textContent = data.metrics.rmse != null ? data.metrics.rmse.toFixed(2) : '—';
                }
                return;
            }
        }
    } catch (e) {
        console.log('Model performance fetch error:', e);
    }

    // Default to un-evaluated state if no metrics file exists
    if (perfStatusPill) {
        perfStatusPill.textContent = 'Pending';
        perfStatusPill.className = 'system-pill demo';
    }
    if (perfEmptyMsg) {
        perfEmptyMsg.classList.remove('hidden');
        perfEmptyMsg.textContent = 'Model evaluation has not been performed yet.';
    }
    if (perfMetricsGrid) perfMetricsGrid.classList.add('hidden');
}

document.addEventListener('DOMContentLoaded', () => {
    checkServerHealth();
    loadHistory();
    loadModelPerformance();
    setInterval(checkServerHealth, 30000);
});

document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') { e.preventDefault(); resetForm(); }
    if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); downloadResults(); }
    if (e.key === 'Escape') {
        fullscreenModal.classList.add('hidden');
        shortcutsModal.classList.add('hidden');
    }
    if (!resultsContainer.classList.contains('hidden') && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
        const delta = e.key === 'ArrowLeft' ? -3 : 3;
        setSliderPosition(sliderPosition + delta);
    }
});
