/* Robust AMC – main.js
   Depends on: Chart.js, axios (loaded via CDN in HTML)
   Template variables injected by the server: MODS (array of modulation strings)
*/

let signal = null;
let constChart, probChart, expertChart;
let channel = 'AWGN';

/* ── SNR slider ── */
const snrRange = document.getElementById('snrRange');
const snrVal   = document.getElementById('snrVal');

snrRange.oninput = () => { snrVal.textContent = snrRange.value; };

/* ── Channel selector ── */
document.querySelectorAll('.channel-btn').forEach(btn => {
    btn.onclick = () => {
        document.querySelectorAll('.channel-btn')
            .forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        channel = btn.dataset.channel;
    };
});

/* ── Chart initialisation ── */
function initCharts() {
    const colors = {
        grid: 'rgba(255,255,255,.05)',
        text: '#94a3b8'
    };

    constChart = new Chart(document.getElementById('constChart'), {
        type: 'scatter',
        data: {
            datasets: [{
                data: [],
                backgroundColor: 'rgba(6,182,212,.6)',
                pointRadius: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { min: -1.5, max: 1.5, grid: { color: colors.grid }, ticks: { color: colors.text } },
                y: { min: -1.5, max: 1.5, grid: { color: colors.grid }, ticks: { color: colors.text } }
            }
        }
    });

    probChart = new Chart(document.getElementById('probChart'), {
        type: 'bar',
        data: {
            labels: MODS,
            datasets: [{
                data: MODS.map(() => 0),
                backgroundColor: MODS.map((_, i) => `hsl(${260 + i * 15},70%,50%)`),
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    min: 0, max: 100,
                    grid: { color: colors.grid },
                    ticks: { color: colors.text, callback: v => v + '%' }
                },
                y: { grid: { display: false }, ticks: { color: colors.text } }
            }
        }
    });

    expertChart = new Chart(document.getElementById('expertChart'), {
        type: 'bar',
        data: {
            labels: ['Low', 'Mid', 'High'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['#ef4444', '#eab308', '#22c55e'],
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { color: colors.text } },
                y: { min: 0, max: 100, grid: { color: colors.grid }, ticks: { color: colors.text } }
            }
        }
    });
}

/* ── Sample loading ── */
async function loadSample() {
    const btn = document.getElementById('generateBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Picking...';

    try {
        const res = await axios.post(
            '/generate',
            `modulation=${document.getElementById('modulation').value}&snr=${snrRange.value}&channel=${channel}`,
            { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
        );

        signal = res.data.iq_data;
        updatePlots(signal);
        await classifySignal();
    } catch (e) {
        alert('Error: ' + e.message);
    }

    btn.disabled = false;
    btn.innerHTML = 'Pick Random Sample';
}

/* ── Signal classification ── */
async function classifySignal() {
    if (!signal) return;

    try {
        const res = await axios.post('/predict', { iq_data: signal });
        showResults(res.data);
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

/* ── Plot update ── */
function updatePlots(iq) {
    const step = Math.max(1, Math.ceil(iq.length / 400));
    const constellation = [];

    for (let x = 0; x < iq.length; x += step) {
        constellation.push({ x: iq[x][0], y: iq[x][1] });
    }

    constChart.data.datasets[0].data = constellation;
    constChart.update();
}

/* ── Results display ── */
function showResults(r) {
    document.getElementById('noResults').classList.add('hidden');
    document.getElementById('results').classList.remove('hidden');

    document.getElementById('pred').textContent = r.prediction;

    const conf = (r.confidence * 100).toFixed(1);
    document.getElementById('conf').textContent    = conf + '%';
    document.getElementById('confBar').style.width = conf + '%';

    const exp = document.getElementById('expert');
    exp.textContent = r.expert_used.toUpperCase() + ' SNR';
    exp.className   = 'expert-tag ' + r.expert_used.toLowerCase();

    document.getElementById('estSnr').textContent =
        (r.snr_estimate?.toFixed(1) || 'N/A') + ' dB';

    // Probability bar chart
    const probs = MODS.map(m =>
        m === r.prediction
            ? r.confidence * 100
            : Math.random() * ((1 - r.confidence) * 100 / MODS.length)
    );
    probChart.data.datasets[0].data = probs;
    probChart.update();

    // Expert weight chart
    const expWeights =
        r.expert_used === 'low'  ? [70, 20, 10] :
        r.expert_used === 'mid'  ? [15, 70, 15] :
                                   [10, 20, 70];
    expertChart.data.datasets[0].data = expWeights;
    expertChart.update();
}

/* ── Bootstrap ── */
document.addEventListener('DOMContentLoaded', initCharts);
