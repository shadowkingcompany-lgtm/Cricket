/**
 * pose.js - MediaPipe Pose estimation & rendering module
 */

let pose = null;
let camera = null;
let activeStream = null;
let lastFrameTime = Date.now();
let isEngineRunning = false;
let isUploadedVideoMode = false;
let isVideoLoopActive = false;
let isMirrored = true;

// UI Elements
const videoEl = document.getElementById('webcamVideo');
const canvasEl = document.getElementById('poseCanvas');
const ctx = canvasEl.getContext('2d');
const btnStart = document.getElementById('btnStartEngine');
const btnStop = document.getElementById('btnStopEngine');
const placeholder = document.getElementById('cameraPlaceholder');
const hud = document.getElementById('hudOverlay');
const latencyDisplay = document.getElementById('latencyDisplay');

// Global state shared with other modules
window.currentLandmarks = null;
window.poseMode = 'batting'; // 'batting' or 'bowling'

// Initialize MediaPipe Pose
function initPose() {
    pose = new Pose({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
        }
    });

    pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        smoothSegmentation: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    pose.onResults(onPoseResults);
}

// MediaPipe Results Callback
function onPoseResults(results) {
    if (!isEngineRunning) return;

    // Calculate latency
    const now = Date.now();
    const latency = now - lastFrameTime;
    lastFrameTime = now;
    latencyDisplay.textContent = `${latency} ms`;

    // Clear and match canvas size to video size
    if (canvasEl.width !== results.image.width || canvasEl.height !== results.image.height) {
        canvasEl.width = results.image.width;
        canvasEl.height = results.image.height;
    }

    ctx.save();
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    // Draw the mirrored video background ONLY if we are in webcam mode
    // (In upload mode, the video is visible directly behind the transparent canvas)
    if (!isUploadedVideoMode) {
        ctx.translate(canvasEl.width, 0);
        ctx.scale(-1, 1);
        ctx.globalAlpha = 0.85; // slightly dim to make skeleton stand out
        ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);
    }
    ctx.restore();
    ctx.globalAlpha = 1.0;

    if (results.poseLandmarks) {
        window.currentLandmarks = results.poseLandmarks;
        
        // Draw Skeleton overlay
        drawSkeleton(results.poseLandmarks);
        
        // Run cricket specific analysis (from analysis.js)
        if (typeof window.analyzePoseFrame === 'function') {
            window.analyzePoseFrame(results.poseLandmarks);
        }
    } else {
        window.currentLandmarks = null;
    }
}

// Draw skeleton with glowing sci-fi style
function drawSkeleton(landmarks) {
    const w = canvasEl.width;
    const h = canvasEl.height;

    // Helper to get canvas coords (mirrored if camera is mirrored)
    function getPt(index) {
        if (!landmarks[index]) return null;
        const xVal = isMirrored ? (1.0 - landmarks[index].x) : landmarks[index].x;
        return {
            x: xVal * w,
            y: landmarks[index].y * h,
            visibility: landmarks[index].visibility
        };
    }

    const connections = [
        // Torso
        [11, 12], [11, 23], [12, 24], [23, 24],
        // Left Arm
        [11, 13], [13, 15],
        // Right Arm
        [12, 14], [14, 16],
        // Left Leg
        [23, 25], [25, 27], [27, 29], [29, 31],
        // Right Leg
        [24, 26], [26, 28], [28, 30], [30, 32]
    ];

    // Determine color scheme based on sports mode
    const isBatting = window.poseMode === 'batting';
    const skeletonColor = isBatting ? 'rgba(0, 242, 254, 0.6)' : 'rgba(57, 255, 20, 0.6)';
    const glowColor = isBatting ? '#00f2fe' : '#39ff14';
    const jointFill = '#ffffff';

    // Draw Limbs/Connections
    ctx.lineWidth = 4;
    ctx.strokeStyle = skeletonColor;
    ctx.shadowBlur = 8;
    ctx.shadowColor = glowColor;

    connections.forEach(([p1, p2]) => {
        const pt1 = getPt(p1);
        const pt2 = getPt(p2);
        
        if (pt1 && pt2 && pt1.visibility > 0.5 && pt2.visibility > 0.5) {
            ctx.beginPath();
            ctx.moveTo(pt1.x, pt1.y);
            ctx.lineTo(pt2.x, pt2.y);
            ctx.stroke();
        }
    });

    // Draw Joints
    ctx.shadowBlur = 12;
    ctx.fillStyle = jointFill;
    ctx.strokeStyle = glowColor;
    ctx.lineWidth = 2;

    const keyJoints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
    keyJoints.forEach(jointIdx => {
        const pt = getPt(jointIdx);
        if (pt && pt.visibility > 0.5) {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 6, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        }
    });

    // Draw custom overlay text labels on canvas for key computed joint angles
    ctx.shadowBlur = 0; // turn off shadow for text legibility
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px Orbitron';
    
    // Check if we have the angles calculated in global scope (set by analysis.js)
    if (window.computedAngles) {
        // Label Front Knee
        const frontKneeJointIdx = isBatting ? 25 : 26; // Left knee for batting right-hand, right knee for bowl
        const kneePt = getPt(frontKneeJointIdx);
        if (kneePt && kneePt.visibility > 0.5 && window.computedAngles.knee) {
            ctx.fillText(`KNEE: ${window.computedAngles.knee}°`, kneePt.x + 12, kneePt.y);
        }

        // Label Backlift Elbow
        const elbowJointIdx = isBatting ? 14 : 13; // Right elbow for batting swing, left for spin/pace bowl
        const elbowPt = getPt(elbowJointIdx);
        if (elbowPt && elbowPt.visibility > 0.5 && window.computedAngles.elbow) {
            ctx.fillText(`ELBOW: ${window.computedAngles.elbow}°`, elbowPt.x + 12, elbowPt.y);
        }

        // Label Spine Tilt
        const shoulderCenterPt = getPt(11);
        if (shoulderCenterPt && shoulderCenterPt.visibility > 0.5 && window.computedAngles.spine) {
            ctx.fillText(`SPINE TILT: ${window.computedAngles.spine}°`, shoulderCenterPt.x + 12, shoulderCenterPt.y - 15);
        }
    }
}

// Start Video & MediaPipe loop
async function startEngine() {
    if (isEngineRunning && !isUploadedVideoMode) return;
    
    // Stop any active uploaded video processing
    await stopEngine();
    
    placeholder.style.display = 'none';
    hud.style.display = 'flex';
    btnStart.disabled = true;
    btnStop.disabled = false;
    
    try {
        isUploadedVideoMode = false;
        isMirrored = true;
        canvasEl.classList.add('mirrored');
        videoEl.classList.add('mirrored');
        videoEl.style.display = 'none';

        // Get media stream
        activeStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                frameRate: { ideal: 30 }
            },
            audio: false
        });
        
        videoEl.srcObject = activeStream;
        
        if (!pose) {
            initPose();
        }
        
        isEngineRunning = true;
        
        // Start loop using MediaPipe Camera class
        camera = new Camera(videoEl, {
            onFrame: async () => {
                if (isEngineRunning && !isUploadedVideoMode) {
                    await pose.send({ image: videoEl });
                }
            },
            width: 640,
            height: 480
        });
        
        await camera.start();
        
        // Trigger save session button activate
        document.getElementById('btnSavePracticeSession').disabled = false;
        
    } catch (err) {
        console.error("Camera access failed: ", err);
        placeholder.innerHTML = `
            <i class="fa-solid fa-triangle-exclamation placeholder-icon neon-text-red"></i>
            <p class="neon-text-red">Webcam Access Blocked or Not Found.</p>
            <p style="font-size: 11px; margin-top: 10px;">Please ensure browser camera permissions are enabled for this address.</p>
        `;
        placeholder.style.display = 'flex';
        btnStart.disabled = false;
        btnStop.disabled = true;
        hud.style.display = 'none';
    }
}

// Stop Video & loop
async function stopEngine() {
    if (!isEngineRunning && !videoEl.src) return;
    
    isEngineRunning = false;
    isVideoLoopActive = false;
    btnStart.disabled = false;
    btnStop.disabled = true;
    placeholder.style.display = 'flex';
    hud.style.display = 'none';
    
    if (camera) {
        await camera.stop();
        camera = null;
    }
    
    if (activeStream) {
        activeStream.getTracks().forEach(track => track.stop());
        activeStream = null;
    }
    
    // Reset video upload parameters
    videoEl.src = '';
    videoEl.srcObject = null;
    videoEl.controls = false;
    videoEl.style.display = 'none';
    
    // Clear canvas
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    
    // Disable save button
    document.getElementById('btnSavePracticeSession').disabled = true;
}

// Mode toggle buttons event handlers
document.getElementById('btnBattingMode').addEventListener('click', () => {
    window.poseMode = 'batting';
    document.getElementById('btnBattingMode').classList.add('active');
    document.getElementById('btnBowlingMode').classList.remove('active');
    
    // Adjust DOM titles
    document.getElementById('strideTitle').textContent = 'Footwork Stride';
    document.getElementById('speedTitle').textContent = 'Swing Speed';
    document.getElementById('gaugeSpeedValue').textContent = '0°/s';
    document.getElementById('gaugeStrideValue').textContent = '0.0m';
    
    // Reset HUD text
    document.getElementById('hudShotText').textContent = 'STANCE';
});

document.getElementById('btnBowlingMode').addEventListener('click', () => {
    window.poseMode = 'bowling';
    document.getElementById('btnBowlingMode').classList.add('active');
    document.getElementById('btnBattingMode').classList.remove('active');
    
    // Adjust DOM titles
    document.getElementById('strideTitle').textContent = 'Release Height';
    document.getElementById('speedTitle').textContent = 'Arm Rot Speed';
    document.getElementById('gaugeSpeedValue').textContent = '0°/s';
    document.getElementById('gaugeStrideValue').textContent = '0.0m';
    
    // Reset HUD text
    document.getElementById('hudShotText').textContent = 'READY';
});

// Attach events
btnStart.addEventListener('click', startEngine);
btnStop.addEventListener('click', stopEngine);

// File Upload processing logic
const fileInput = document.getElementById('videoUploadInput');

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Stop active camera engine if running
    await stopEngine();

    // Set states
    isUploadedVideoMode = true;
    isMirrored = false;

    // Reset mirroring classes
    videoEl.classList.remove('mirrored');
    canvasEl.classList.remove('mirrored');

    // Create object URL and bind to player
    const fileUrl = URL.createObjectURL(file);
    videoEl.src = fileUrl;
    videoEl.srcObject = null;
    videoEl.controls = true;
    videoEl.loop = true;
    videoEl.autoplay = true;

    // Display elements
    placeholder.style.display = 'none';
    hud.style.display = 'flex';
    videoEl.style.display = 'block';

    // Button states
    btnStart.disabled = false; // allow switching back to camera
    btnStop.disabled = false;

    if (!pose) {
        initPose();
    }
    isEngineRunning = true;

    // Enable save button
    document.getElementById('btnSavePracticeSession').disabled = false;
});

// Frame processing loop for video element
async function processVideoFrame() {
    if (!isEngineRunning || !isUploadedVideoMode) {
        isVideoLoopActive = false;
        return;
    }
    
    if (!videoEl.paused && !videoEl.ended) {
        try {
            await pose.send({ image: videoEl });
        } catch (err) {
            console.error("MediaPipe Pose video send failed:", err);
        }
        requestAnimationFrame(processVideoFrame);
    } else {
        isVideoLoopActive = false;
    }
}

videoEl.addEventListener('play', () => {
    if (isEngineRunning && isUploadedVideoMode && !isVideoLoopActive) {
        isVideoLoopActive = true;
        processVideoFrame();
    }
});

videoEl.addEventListener('seeked', async () => {
    if (isEngineRunning && isUploadedVideoMode) {
        try {
            await pose.send({ image: videoEl });
        } catch (err) {
            console.error("MediaPipe Pose video seek send failed:", err);
        }
    }
});

videoEl.addEventListener('loadeddata', async () => {
    if (isEngineRunning && isUploadedVideoMode) {
        try {
            await pose.send({ image: videoEl });
        } catch (err) {
            console.error("MediaPipe Pose video initial send failed:", err);
        }
    }
});

// Initialize MediaPipe once DOM loads
window.addEventListener('DOMContentLoaded', () => {
    initPose();
});
