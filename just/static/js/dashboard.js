/**
 * dashboard.js - Main Application Controller
 * Handles routing, session logging, profile synchronization,
 * radar charts, and ML prediction API integration.
 */

let radarChart = null;

// Initialize Dashboard
window.addEventListener('DOMContentLoaded', () => {
    initRouting();
    initProfile();
    initSessionsTable();
    initRadarChart();
    initPredictorSliders();
});

// 1. Sidebar Tab Navigation Routing
function initRouting() {
    const navItems = document.querySelectorAll('.nav-menu .nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const tabId = item.getAttribute('data-tab');

            // Set active sidebar item
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            // Set active tab content
            tabContents.forEach(tab => tab.classList.remove('active'));
            const targetTab = document.getElementById(`${tabId}-tab`);
            if (targetTab) targetTab.classList.add('active');

            // Update page header
            pageTitle.textContent = item.querySelector('span').textContent;
            
            // If switching to NN Progress Model, trigger canvas resize to fit bounding box
            if (tabId === 'progress-tracker' && typeof resizeNNCanvas === 'function') {
                setTimeout(resizeNNCanvas, 100);
            }
        });
    });
}

// 2. Profile Management
async function initProfile() {
    const form = document.getElementById('profileForm');
    
    // Fetch initial profile from Flask DB
    try {
        const response = await fetch('/api/profile');
        const profile = await response.json();
        
        updateProfileDOM(profile);
        
        // Fill form fields
        document.getElementById('profileName').value = profile.name;
        document.getElementById('profileRole').value = profile.role;
        document.getElementById('profileExperience').value = profile.experience;
        document.getElementById('profileBattingHand').value = profile.battingHand;
        document.getElementById('profileBowlingStyle').value = profile.bowlingStyle;
    } catch (err) {
        console.error("Failed to load profile:", err);
    }

    // Submit form handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const updatedProfile = {
            name: document.getElementById('profileName').value,
            role: document.getElementById('profileRole').value,
            experience: document.getElementById('profileExperience').value,
            battingHand: document.getElementById('profileBattingHand').value,
            bowlingStyle: document.getElementById('profileBowlingStyle').value
        };

        try {
            const response = await fetch('/api/profile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updatedProfile)
            });
            
            const result = await response.json();
            if (result.status === 'success') {
                updateProfileDOM(result.profile);
                alert("Player Profile successfully synchronized with AI Core.");
            }
        } catch (err) {
            console.error("Failed to update profile:", err);
            alert("Error syncing profile to server.");
        }
    });
}

function updateProfileDOM(profile) {
    document.getElementById('profileNameDisplay').textContent = profile.name;
    document.getElementById('profileRoleDisplay').textContent = profile.role;
    
    // Avatar Initial
    const avatar = document.getElementById('avatarNameInit');
    if (avatar && profile.name) {
        avatar.textContent = profile.name.charAt(0).toUpperCase();
    }
}

// 3. Practice Sessions Sync
async function initSessionsTable() {
    try {
        const response = await fetch('/api/sessions');
        const sessions = await response.json();
        
        populateSessionsTable(sessions);
        updateRadarValues(sessions);
    } catch (err) {
        console.error("Failed to fetch sessions database:", err);
    }
}

function populateSessionsTable(sessions) {
    const tbody = document.querySelector('#sessionTable tbody');
    tbody.innerHTML = '';

    if (sessions.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-item">
                <td colspan="8" style="text-align: center; color: var(--color-text-muted); font-style: italic;">
                    No practice logs found. Start the AI webcam analyzer to record telemetry.
                </td>
            </tr>
        `;
        return;
    }

    // Reverse chronological order
    const sorted = [...sessions].reverse();

    sorted.forEach(session => {
        const tr = document.createElement('tr');
        
        const isBatting = session.type === 'batting';
        const metric1Name = isBatting ? 'Stability' : 'Speed';
        const metric1Val = isBatting ? `${session.metric1}%` : `${session.metric1}°/s`;

        const metric2Name = isBatting ? 'Stride' : 'Rel Ht';
        const metric2Val = isBatting ? `${(session.metric2).toFixed(2)}m` : `${(session.metric2).toFixed(2)}m`;

        const metric3Name = isBatting ? 'Swing' : 'Arm Rot';
        const metric3Val = isBatting ? `${session.metric3}°/s` : `${session.metric3}°/s`;

        tr.innerHTML = `
            <td>#${session.id}</td>
            <td style="font-family: var(--font-tech);">${session.date}</td>
            <td><span class="badge ${isBatting ? 'badge-teal' : 'badge-green'}">${session.type}</span></td>
            <td style="font-weight: 600;">${session.shot}</td>
            <td>${metric1Val} <span style="font-size: 8px; color: var(--color-text-muted); block">${metric1Name}</span></td>
            <td>${metric2Val} <span style="font-size: 8px; color: var(--color-text-muted); block">${metric2Name}</span></td>
            <td>${metric3Val} <span style="font-size: 8px; color: var(--color-text-muted); block">${metric3Name}</span></td>
            <td style="font-family: var(--font-tech); font-weight: 700;" class="neon-text-${isBatting ? 'teal' : 'green'}">${session.score}%</td>
        `;
        
        tbody.appendChild(tr);
    });
}

// Log practice sessions gathered in the UI from CV
const btnSaveSession = document.getElementById('btnSavePracticeSession');
btnSaveSession.addEventListener('click', async () => {
    if (window.localSessionDetections.length === 0) {
        alert("No actions detected. Perform shots/bowls in front of the camera first!");
        return;
    }

    btnSaveSession.disabled = true;
    let savedCount = 0;

    // Post each locally buffered session detection to database
    for (const det of window.localSessionDetections) {
        const payload = {
            date: new Date().toISOString().split('T')[0],
            type: det.type,
            shot: det.action,
            // Map metrics back depending on batting vs bowling
            metric1: det.type === 'batting' ? det.val3 : det.val1, // Batting: Stability, Bowling: Speed
            metric2: det.val2, // Batting: Stride, Bowling: Release Height
            metric3: det.type === 'batting' ? det.val1 : det.val1, // Batting: Swing Speed, Bowling: Arm Rotation Speed
            score: det.score
        };

        try {
            const response = await fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            if (result.status === 'success') {
                savedCount++;
            }
        } catch (err) {
            console.error("Session log error:", err);
        }
    }

    alert(`Successfully synchronized ${savedCount} bio-metric session(s) to cloud database.`);
    
    // Clear buffer & reset
    window.localSessionDetections = [];
    document.getElementById('liveSessionLogger').innerHTML = '<li class="empty-item">No actions detected yet. Start practice.</li>';
    
    // Reload table & update analytics
    await initSessionsTable();
});

// 4. Radar Weakness Chart initialization (Chart.js)
function initRadarChart() {
    const ctx = document.getElementById('radarChart');
    if (!ctx) return;

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Footwork & Stride', 'Swing Speed', 'Core Balance', 'Arm Rotation Speed', 'Stance Stability'],
            datasets: [{
                label: 'Biometric Alignment Score',
                data: [70, 75, 80, 70, 85], // default values
                backgroundColor: 'rgba(0, 242, 254, 0.15)',
                borderColor: 'var(--neon-teal)',
                borderWidth: 2,
                pointBackgroundColor: 'var(--neon-magenta)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'var(--neon-magenta)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: {
                        color: '#94a3b8',
                        font: { family: 'Orbitron', size: 10 }
                    },
                    ticks: {
                        backdropColor: 'transparent',
                        color: 'rgba(255, 255, 255, 0.5)',
                        font: { size: 8 }
                    },
                    min: 0,
                    max: 100
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Calculate dynamic weakness radar ratings based on actual session scores
function updateRadarValues(sessions) {
    if (!radarChart || sessions.length === 0) return;

    // Filter batting and bowling
    const batting = sessions.filter(s => s.type === 'batting');
    const bowling = sessions.filter(s => s.type === 'bowling');

    // Default parameters
    let strideScore = 70;
    let swingScore = 72;
    let balanceScore = 78;
    let bowlingSpeedScore = 65;
    let stanceScore = 80;

    if (batting.length > 0) {
        const avgScore = batting.reduce((acc, s) => acc + s.score, 0) / batting.length;
        const avgStride = batting.reduce((acc, s) => acc + s.metric2, 0) / batting.length;
        const avgSpeed = batting.reduce((acc, s) => acc + s.metric3, 0) / batting.length;
        const avgStability = batting.reduce((acc, s) => acc + s.metric1, 0) / batting.length;
        
        strideScore = Math.min(100, Math.round((avgStride / 1.4) * 100));
        swingScore = Math.min(100, Math.round((avgSpeed / 800) * 100));
        stanceScore = Math.min(100, Math.round(avgStability));
    }

    if (bowling.length > 0) {
        const avgBowlSpeed = bowling.reduce((acc, s) => acc + s.metric1, 0) / bowling.length;
        const avgHeight = bowling.reduce((acc, s) => acc + s.metric2, 0) / bowling.length;
        
        bowlingSpeedScore = Math.min(100, Math.round((avgBowlSpeed / 900) * 100));
        balanceScore = Math.min(100, Math.round((avgHeight / 2.3) * 100));
    }

    // Apply values to chart
    radarChart.data.datasets[0].data = [strideScore, swingScore, balanceScore, bowlingSpeedScore, stanceScore];
    
    // Set colors based on player profile primary role
    const role = document.getElementById('profileRoleDisplay').textContent.toLowerCase();
    if (role.includes('bowler')) {
        radarChart.data.datasets[0].borderColor = 'var(--neon-green)';
        radarChart.data.datasets[0].backgroundColor = 'rgba(57, 255, 20, 0.15)';
    } else {
        radarChart.data.datasets[0].borderColor = 'var(--neon-teal)';
        radarChart.data.datasets[0].backgroundColor = 'rgba(0, 242, 254, 0.15)';
    }
    
    radarChart.update();
}

// 5. Match Performance Predictor Sliders
function initPredictorSliders() {
    const sliders = [
        { id: 'slideAvgScore', displayId: 'valAvgScore', suffix: '%' },
        { id: 'slideRecentRuns', displayId: 'valRecentRuns', suffix: '' },
        { id: 'slideRecentSR', displayId: 'valRecentSR', suffix: '' },
        { id: 'slideRecentWickets', displayId: 'valRecentWickets', suffix: '' },
        { id: 'slideTrainingHours', displayId: 'valTrainingHours', suffix: ' hrs' }
    ];

    sliders.forEach(slide => {
        const el = document.getElementById(slide.id);
        const display = document.getElementById(slide.displayId);
        
        el.addEventListener('input', () => {
            display.textContent = el.value + slide.suffix;
        });
    });

    // Run prediction action
    document.getElementById('btnRunPrediction').addEventListener('click', runMLPrediction);
}

// Trigger POST request to backend Flask ML regression model
async function runMLPrediction() {
    const status = document.getElementById('predStatus');
    status.textContent = "COMPUTING...";
    status.className = "badge badge-magenta animate-pulse";

    const payload = {
        avgScore: parseInt(document.getElementById('slideAvgScore').value),
        recentRuns: parseInt(document.getElementById('slideRecentRuns').value),
        recentSR: parseInt(document.getElementById('slideRecentSR').value),
        recentWickets: parseFloat(document.getElementById('slideRecentWickets').value),
        trainingHours: parseInt(document.getElementById('slideTrainingHours').value)
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        
        if (data.status === 'success') {
            const preds = data.predictions;
            const weights = data.featureImportance;

            // Render Output values with numerical counting animation
            animateNumber('predRuns', preds.runs);
            animateNumber('predSR', preds.strikeRate);
            animateNumber('predWickets', preds.wickets);
            animateNumber('predEconomy', preds.economy);
            animateNumber('predImpact', preds.impact);

            // Update impact rating progress bar width
            document.getElementById('predImpactBar').style.width = `${preds.impact * 10}%`;

            // Populate ML Weights list explanation
            populateWeightsList(weights);

            status.textContent = "CONVERGED";
            status.className = "badge badge-magenta";
        }
    } catch (err) {
        console.error("ML Prediction connection error:", err);
        status.textContent = "ERROR";
        status.className = "badge badge-danger";
    }
}

// Visual counting animation helper
function animateNumber(id, endVal) {
    const el = document.getElementById(id);
    const startVal = parseFloat(el.textContent) || 0;
    const duration = 800; // ms
    const startTime = performance.now();

    function update(time) {
        const elapsed = time - startTime;
        const progress = Math.min(1, elapsed / duration);
        const current = startVal + (endVal - startVal) * progress;
        
        // Match decimal place
        el.textContent = endVal % 1 === 0 ? Math.round(current) : current.toFixed(1);

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            el.textContent = endVal;
        }
    }
    requestAnimationFrame(update);
}

function populateWeightsList(weights) {
    const list = document.getElementById('featureWeightsList');
    list.innerHTML = '';

    // Sort features by absolute weight strength
    const sortedFeatures = Object.entries(weights).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

    sortedFeatures.forEach(([feature, weight]) => {
        const li = document.createElement('li');
        const isPositive = weight >= 0;
        
        li.innerHTML = `
            <span style="font-weight: 500;">${feature}</span>
            <span class="badge ${isPositive ? 'badge-teal' : 'badge-outline'}" style="font-family: var(--font-tech);">
                ${isPositive ? '+' : ''}${weight.toFixed(3)}
            </span>
        `;
        list.appendChild(li);
    });
}
