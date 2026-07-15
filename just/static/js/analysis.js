/**
 * analysis.js - Cricket Biomechanics Biometric Engine
 * Analyzes pose landmarks to calculate joint angles, speed, footwork,
 * and detects cricket shots/bowling actions.
 */

// Global storage for angles to display on canvas overlay
window.computedAngles = {
    knee: 180,
    elbow: 180,
    spine: 0
};

// State variables for Shot/Bowling action detection
let swingActive = false;
let maxSwingSpeed = 0;
let maxFootStride = 0;
let minKneeAngle = 180;
let maxElbowAngle = 0;
let maxReleaseHeight = 0;
let maxArmRotSpeed = 0;

let lastHandPos = null;
let lastArmAngle = null;
let lastFrameTimeAnalysis = Date.now();

// Buffer to store recent session detections locally before saving to server
window.localSessionDetections = [];

// Helper: Calculate 2D angle between three points (A, B, C) with B as vertex
function calculateAngle(a, b, c) {
    if (!a || !b || !c) return 180;
    
    // Vector BA
    const ba = { x: a.x - b.x, y: a.y - b.y };
    // Vector BC
    const bc = { x: c.x - b.x, y: c.y - b.y };
    
    // Dot product
    const dotProduct = ba.x * bc.x + ba.y * bc.y;
    // Magnitudes
    const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y);
    const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);
    
    if (magBA * magBC === 0) return 180;
    
    const cosAngle = dotProduct / (magBA * magBC);
    // Clamp to avoid float precision issues outside [-1, 1]
    const clampedCos = Math.max(-1, Math.min(1, cosAngle));
    
    const angleRad = Math.acos(clampedCos);
    return Math.round(angleRad * 180 / Math.PI);
}

// Helper: Calculate Euclidean distance
function getDistance(pt1, pt2) {
    if (!pt1 || !pt2) return 0;
    return Math.sqrt(Math.pow(pt1.x - pt2.x, 2) + Math.pow(pt1.y - pt2.y, 2));
}

/**
 * Main entry point called by pose.js on every pose estimation result
 * @param {Array} landmarks - MediaPipe pose landmarks
 */
window.analyzePoseFrame = function(landmarks) {
    const isBatting = window.poseMode === 'batting';
    
    // 1. EXTRACT RELEVANT LANDMARKS
    // MediaPipe landmark indices:
    // Left Shoulder: 11, Right Shoulder: 12
    // Left Elbow: 13, Right Elbow: 14
    // Left Wrist: 15, Right Wrist: 16
    // Left Hip: 23, Right Hip: 24
    // Left Knee: 25, Right Knee: 26
    // Left Ankle: 27, Right Ankle: 28
    // Left Foot: 31, Right Foot: 32
    
    const lShoulder = landmarks[11];
    const rShoulder = landmarks[12];
    const lElbow = landmarks[13];
    const rElbow = landmarks[14];
    const lWrist = landmarks[15];
    const rWrist = landmarks[16];
    const lHip = landmarks[23];
    const rHip = landmarks[24];
    const lKnee = landmarks[25];
    const rKnee = landmarks[26];
    const lAnkle = landmarks[27];
    const rAnkle = landmarks[28];
    
    // Check key visibilities
    if (!lShoulder || !rShoulder || !lHip || !rHip || lShoulder.visibility < 0.5 || rShoulder.visibility < 0.5) {
        return; // Landmarks not clear enough
    }
    
    const now = Date.now();
    const dt = (now - lastFrameTimeAnalysis) / 1000.0;
    lastFrameTimeAnalysis = now;
    if (dt <= 0) return;

    // 2. BIOMECHANICAL ANGLE COMPUTATION
    let kneeAngle = 180;
    let elbowAngle = 180;
    let spineTilt = 0;
    
    if (isBatting) {
        // For right-handed batsman (default profile), left knee is the front knee (facing bowler)
        kneeAngle = calculateAngle(lHip, lKnee, lAnkle);
        // Backlift elbow (right elbow is the back arm generating swing)
        elbowAngle = calculateAngle(rShoulder, rElbow, rWrist);
    } else {
        // Bowling mode: Front knee is right knee (right-arm fast bowler landing stride)
        kneeAngle = calculateAngle(rHip, rKnee, rAnkle);
        // Bowling arm elbow angle (left/right depending on action)
        elbowAngle = calculateAngle(lShoulder, lElbow, lWrist);
    }
    
    // Spine Tilt: Angle of shoulder line relative to vertical
    // Vector between shoulders
    const shoulderDx = rShoulder.x - lShoulder.x;
    const shoulderDy = rShoulder.y - lShoulder.y;
    spineTilt = Math.abs(Math.round(Math.atan2(shoulderDy, shoulderDx) * 180 / Math.PI));
    if (spineTilt > 90) spineTilt = 180 - spineTilt;

    // Update global variables for canvas drawings
    window.computedAngles = {
        knee: kneeAngle,
        elbow: elbowAngle,
        spine: spineTilt
    };

    // Update angle sliders/bars in UI
    document.getElementById('valFrontKnee').textContent = `${kneeAngle}°`;
    document.getElementById('barFrontKnee').style.width = `${Math.min(100, (kneeAngle/180)*100)}%`;
    
    document.getElementById('valElbow').textContent = `${elbowAngle}°`;
    document.getElementById('barElbow').style.width = `${Math.min(100, (elbowAngle/180)*100)}%`;
    
    document.getElementById('valSpine').textContent = `${spineTilt}°`;
    document.getElementById('barSpine').style.width = `${Math.min(100, (spineTilt/60)*100)}%`;

    // 3. TELEMETRY COMPUTATION (DYNAMIC UPDATES)
    
    // Average Hip height stability represents Stance Stability
    const hipHeight = (lHip.y + rHip.y) / 2.0;
    // Compare current hip height stability (high stability = low frame deviation)
    let stanceStability = 95 - Math.min(45, Math.abs(hipHeight - 0.7) * 200); // normalized stabilizer score
    stanceStability = Math.max(50, Math.min(100, Math.round(stanceStability)));
    
    // Update Stance Stability Gauge
    document.getElementById('gaugeStanceValue').textContent = `${stanceStability}%`;
    const stanceDash = Math.max(0, 251 - (251 * stanceStability) / 100);
    document.getElementById('gaugeStanceMeter').style.strokeDashoffset = stanceDash;

    if (isBatting) {
        // --- BATTING TELEMETRY ---
        // Stride: horizontal ankle separation scaled by shoulder width (calibration)
        const shoulderWidth = getDistance(lShoulder, rShoulder);
        const ankleSeparation = Math.abs(lAnkle.x - rAnkle.x);
        const strideMeters = shoulderWidth > 0 ? (ankleSeparation / shoulderWidth) * 0.45 : 0; // approximate physical metric
        const stridePercent = Math.min(100, (strideMeters / 1.5) * 100);

        document.getElementById('gaugeStrideValue').textContent = `${strideMeters.toFixed(2)}m`;
        const strideDash = Math.max(0, 251 - (251 * stridePercent) / 100);
        document.getElementById('gaugeStrideMeter').style.strokeDashoffset = strideDash;

        // Wrist speed (swing velocity)
        const rWristPos = { x: rWrist.x, y: rWrist.y };
        let swingSpeedDegSec = 0;
        
        if (lastHandPos) {
            const handDist = getDistance(rWristPos, lastHandPos);
            // Hand speed in coordinate units/sec
            const speedCoeff = handDist / dt;
            swingSpeedDegSec = Math.round(speedCoeff * 550); // Scale to cricket bat degrees/second
        }
        lastHandPos = rWristPos;

        // Limit speed jitter
        swingSpeedDegSec = Math.min(850, swingSpeedDegSec);
        
        document.getElementById('gaugeSpeedValue').textContent = `${swingSpeedDegSec}°/s`;
        const speedPercent = Math.min(100, (swingSpeedDegSec / 800) * 100);
        const speedDash = Math.max(0, 251 - (251 * speedPercent) / 100);
        document.getElementById('gaugeSpeedMeter').style.strokeDashoffset = speedDash;

        // --- BATTING STATE MACHINE / SHOT DETECTION ---
        // Thresholds
        const SWING_START_THRESHOLD = 280; // Speed to trigger swing
        const SWING_END_THRESHOLD = 150;
        
        if (swingSpeedDegSec > SWING_START_THRESHOLD) {
            if (!swingActive) {
                swingActive = true;
                maxSwingSpeed = 0;
                maxFootStride = 0;
                minKneeAngle = 180;
                maxElbowAngle = 0;
                document.getElementById('hudCalibrationText').textContent = "CALCULATING";
                document.getElementById('hudCalibrationText').parentElement.className = "hud-item calibration warning";
            }
            
            // Record peak metrics during active swing phase
            if (swingSpeedDegSec > maxSwingSpeed) maxSwingSpeed = swingSpeedDegSec;
            if (strideMeters > maxFootStride) maxFootStride = strideMeters;
            if (kneeAngle < minKneeAngle) minKneeAngle = kneeAngle;
            if (elbowAngle > maxElbowAngle) maxElbowAngle = elbowAngle;
            
        } else if (swingActive && swingSpeedDegSec < SWING_END_THRESHOLD) {
            // Swing completed! Classify the shot.
            swingActive = false;
            document.getElementById('hudCalibrationText').textContent = "OK";
            document.getElementById('hudCalibrationText').parentElement.className = "hud-item calibration";
            
            classifyBattingShot(maxSwingSpeed, maxFootStride, minKneeAngle, maxElbowAngle, stanceStability);
        }

    } else {
        // --- BOWLING TELEMETRY ---
        // Release Height: Height of bowling wrist (e.g. Right Wrist) relative to right shoulder
        const releaseHtNormalized = Math.max(0, rShoulder.y - rWrist.y) * 2.5; // offset above shoulder
        const releaseHtMeters = 1.8 + releaseHtNormalized; // scale base height of standard bowler
        const releasePercent = Math.min(100, (releaseHtNormalized / 0.8) * 100);
        
        document.getElementById('gaugeStrideValue').textContent = `${releaseHtMeters.toFixed(2)}m`;
        const releaseDash = Math.max(0, 251 - (251 * releasePercent) / 100);
        document.getElementById('gaugeStrideMeter').style.strokeDashoffset = releaseDash;

        // Bowling Arm Rotation Speed (Angular Velocity of Right Arm)
        // Vector from Right Shoulder (12) to Right Wrist (16)
        const armDx = rWrist.x - rShoulder.x;
        const armDy = rWrist.y - rShoulder.y;
        const armAngle = Math.atan2(armDy, armDx); // angle in radians
        let armRotSpeedDegSec = 0;

        if (lastArmAngle !== null) {
            let angleDiff = armAngle - lastArmAngle;
            // Normalize angle diff to [-PI, PI] to handle wrapping
            if (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
            if (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;
            
            armRotSpeedDegSec = Math.abs(Math.round((angleDiff / dt) * (180 / Math.PI)));
        }
        lastArmAngle = armAngle;

        // Apply visual filtering
        armRotSpeedDegSec = Math.min(950, armRotSpeedDegSec);

        document.getElementById('gaugeSpeedValue').textContent = `${armRotSpeedDegSec}°/s`;
        const armSpeedPercent = Math.min(100, (armRotSpeedDegSec / 900) * 100);
        const armSpeedDash = Math.max(0, 251 - (251 * armSpeedPercent) / 100);
        document.getElementById('gaugeSpeedMeter').style.strokeDashoffset = armSpeedDash;

        // --- BOWLING STATE MACHINE / DETECTION ---
        const BOWL_START_THRESHOLD = 300;
        const BOWL_END_THRESHOLD = 120;

        if (armRotSpeedDegSec > BOWL_START_THRESHOLD) {
            if (!swingActive) {
                swingActive = true;
                maxArmRotSpeed = 0;
                maxReleaseHeight = 0;
                minKneeAngle = 180;
                document.getElementById('hudCalibrationText').textContent = "DELIVERING";
                document.getElementById('hudCalibrationText').parentElement.className = "hud-item calibration warning";
            }

            if (armRotSpeedDegSec > maxArmRotSpeed) maxArmRotSpeed = armRotSpeedDegSec;
            if (releaseHtMeters > maxReleaseHeight) maxReleaseHeight = releaseHtMeters;
            if (kneeAngle < minKneeAngle) minKneeAngle = kneeAngle;

        } else if (swingActive && armRotSpeedDegSec < BOWL_END_THRESHOLD) {
            swingActive = false;
            document.getElementById('hudCalibrationText').textContent = "OK";
            document.getElementById('hudCalibrationText').parentElement.className = "hud-item calibration";
            
            classifyBowlingAction(maxArmRotSpeed, maxReleaseHeight, minKneeAngle, stanceStability);
        }
    }
};

/**
 * Heuristic classifier for Batting Shots
 */
function classifyBattingShot(speed, stride, knee, elbow, stability) {
    let shotName = "UNKNOWN SWING";
    let score = 65;
    
    // Heuristic Rules
    if (stride > 0.8 && knee < 125) {
        if (knee < 105) {
            shotName = "SWEEP SHOT";
            // Sweep score: Needs low knee bend (deep squat), stable torso
            score = Math.round(70 + (105 - knee) * 1.5 + (stability - 75) * 0.4);
        } else {
            shotName = "COVER DRIVE";
            // Cover Drive score: Needs good stride length, elbow bend, high stability
            score = Math.round(65 + (stride - 0.7) * 45 + (stability - 70) * 0.5);
        }
    } else if (stride < 0.65 && speed < 320 && knee > 145) {
        shotName = "DEFENSIVE BLOCK";
        // Defense score: Needs compact stride and very high stance stability
        score = Math.round(75 + (stability - 75) * 0.8 - (stride - 0.4) * 20);
    } else if (speed > 450 && stride > 0.6) {
        shotName = "PULL SHOT";
        // Pull score: Needs high swing speed
        score = Math.round(70 + (speed - 450) * 0.08 + (stability - 70) * 0.3);
    } else {
        shotName = "LOFTED DRIVE";
        score = Math.round(60 + (speed - 250) * 0.05 + (stability - 70) * 0.4);
    }
    
    score = Math.max(50, Math.min(99, score));
    triggerShotBanner(shotName, score, speed, stride);
    
    // Add to session log
    addSessionToBuffer({
        type: 'batting',
        action: shotName,
        val1: speed, // swing speed
        val2: stride, // stride extension
        val3: stability, // stability index
        score: score
    });
}

/**
 * Heuristic classifier for Bowling Actions
 */
function classifyBowlingAction(speed, height, knee, stability) {
    let actionName = "PACER DELIVERY";
    let score = 70;
    
    // Heuristics: Spin bowling has lower speed, higher vertical alignment,
    // Fast bowling has higher angular velocity and lower knee angle (lunge flex)
    if (speed > 480) {
        actionName = "FAST BOWLING";
        // Score based on arm speed and delivery release height
        score = Math.round(60 + (speed - 480) * 0.06 + (height - 1.8) * 35);
    } else {
        actionName = "SPIN BOWLING";
        // Score based on balance index (stability) and consistent high release height
        score = Math.round(65 + (stability - 70) * 0.6 + (height - 1.9) * 40);
    }
    
    score = Math.max(50, Math.min(99, score));
    triggerShotBanner(actionName, score, speed, height);
    
    // Add to session log
    addSessionToBuffer({
        type: 'bowling',
        action: actionName,
        val1: speed, // arm speed
        val2: height, // release height
        val3: stability, // stability index
        score: score
    });
}

// Display detected shot inside HUD with animation
function triggerShotBanner(name, score, metric1, metric2) {
    const textEl = document.getElementById('hudShotText');
    textEl.textContent = `${name} [${score}%]`;
    
    // Flash HUD border
    const hudContainer = document.getElementById('hudOverlay');
    hudContainer.style.borderColor = window.poseMode === 'batting' ? 'var(--neon-teal)' : 'var(--neon-green)';
    hudContainer.style.boxShadow = window.poseMode === 'batting' ? '0 0 15px var(--neon-teal-glow)' : '0 0 15px var(--neon-green-glow)';
    
    setTimeout(() => {
        hudContainer.style.borderColor = 'var(--border-color)';
        hudContainer.style.boxShadow = '0 4px 15px rgba(0,0,0,0.5)';
    }, 1500);
}

// Append logs to user list on page
function addSessionToBuffer(data) {
    const list = document.getElementById('liveSessionLogger');
    
    // Remove empty placeholder
    const empty = list.querySelector('.empty-item');
    if (empty) {
        list.innerHTML = '';
    }
    
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const li = document.createElement('li');
    
    const metricStr = data.type === 'batting' 
        ? `${data.val1}°/s, Stride: ${data.val2.toFixed(1)}m` 
        : `${data.val1}°/s, Rel Ht: ${data.val2.toFixed(1)}m`;

    li.innerHTML = `
        <span class="log-time">${timeStr}</span>
        <span class="log-msg neon-text-${data.type === 'batting' ? 'teal' : 'green'}">${data.action}</span>
        <span style="font-size: 10px; color: var(--color-text-muted);">${metricStr}</span>
        <span class="badge ${data.type === 'batting' ? 'badge-teal' : 'badge-green'}">${data.score}%</span>
    `;
    
    list.prepend(li);
    
    // Store in global array to post to database later
    window.localSessionDetections.push(data);
}
