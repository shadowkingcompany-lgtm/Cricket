/**
 * coach.js - AI Coach Chat Assistant & Drill Center
 * Handles chatbot interface and personalized instruction recommendation engine.
 */

const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInputInput');
const btnSend = document.getElementById('btnChatSend');

// Dialog dataset for responses based on keyword analysis
const COACH_DRILLS = {
    "cover drive": {
        drillName: "Front Foot Cover Drive Shadow Batting",
        duration: "20 mins",
        priority: "CRITICAL",
        target: "Stride Extension >= 1.25m",
        instructions: "Place a marker 1.2m ahead of stance. Practice striding front foot completely to marker while executing shadow Cover Drive swings, aligning shoulder to ball path.",
        coachTip: "Your left knee bend is crucial! Lowering your hips during the drive gives you more stability. Focus on pushing the weight off your back foot."
    },
    "pull shot": {
        drillName: "Horizontal Sweep and Twist Drill",
        duration: "15 mins",
        priority: "RECOMMENDED",
        target: "Swing Speed >= 550°/s",
        instructions: "Stand in stance and have a partner throw light bounce balls to chest height. Pivot on your back foot, clear your front hip, and sweep the bat horizontally, finishing high behind the left shoulder.",
        coachTip: "Keep your chest high and eyes locked on the ball. Do not look away. The power comes from hip rotation!"
    },
    "sweep shot": {
        drillName: "Low stance Sweep Alignment Drill",
        duration: "15 mins",
        priority: "RECOMMENDED",
        target: "Knee Angle <= 100° (deep lunge)",
        instructions: "Get down on one knee (deep lunge stance). Swing bat low, parallel to ground, brushing grass surface. Pivot torso fully.",
        coachTip: "Sweeping requires you to get low early. If you bend your front knee properly, it protects you from top-edging the ball into your face."
    },
    "defensive block": {
        drillName: "High Elbow Mirror Shadow Defense",
        duration: "15 mins",
        priority: "RECOMMENDED",
        target: "Stance Stability > 85%",
        instructions: "Conduct 30 defensive blocks in front of a mirror. Focus on maintaining a vertical bat line, high top-hand elbow, and compact stance with front knee bent.",
        coachTip: "A solid defense is the foundation of long innings. Make sure the bat face points down to ensure the ball is pushed straight into the ground."
    },
    "fast bowling": {
        drillName: "Kinetic Arm Rotation Speed Drill",
        duration: "25 mins",
        priority: "CRITICAL",
        target: "Arm Rot Speed >= 600°/s",
        instructions: "Stand 4 steps away from delivery stride. Run up, focus on a high knee lift, strong front-arm pull down, and explosive rotation of the bowling arm over your shoulder.",
        coachTip: "Power starts from the ground up. Sticking your front leg straight during landing creates a pivot block that increases arm velocity."
    },
    "spin bowling": {
        drillName: "Wrist Flick Release Drill",
        duration: "20 mins",
        priority: "RECOMMENDED",
        target: "Release Height >= 2.0m",
        instructions: "Stand close to a wall. Practice bowling spin deliveries focusing purely on index/middle finger release flicks, causing the ball to spin off the wall.",
        coachTip: "A high release point is essential for loop and dip. Make sure your wrist finishes high and flips towards the target."
    }
};

// Send message functions
function sendUserMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    // Add User Bubble
    appendBubble('user', text, 'U');
    chatInput.value = '';
    
    // Scroll down
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Typing indicator
    const typingBubble = appendTypingIndicator();
    
    // Process response after delay
    setTimeout(() => {
        typingBubble.remove();
        generateCoachResponse(text);
    }, 1000 + Math.random() * 800);
}

function appendBubble(sender, text, avatarChar) {
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${sender}`;
    
    bubble.innerHTML = `
        <div class="chat-avatar">${avatarChar}</div>
        <div class="chat-text">
            <p>${text}</p>
        </div>
    `;
    
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
}

function appendTypingIndicator() {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble coach typing-indicator';
    bubble.innerHTML = `
        <div class="chat-avatar">C</div>
        <div class="chat-text" style="display:flex; gap: 4px; padding: 12px 18px;">
            <span class="dot-pulse" style="animation: pulse 1s infinite alternate;">.</span>
            <span class="dot-pulse" style="animation: pulse 1s infinite alternate 0.2s;">.</span>
            <span class="dot-pulse" style="animation: pulse 1s infinite alternate 0.4s;">.</span>
        </div>
    `;
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
}

// Generate Coach Response based on input keywords
function generateCoachResponse(input) {
    const cleanInput = input.toLowerCase();
    let responseText = "";
    let drillKeyword = null;

    // Detect keywords
    if (cleanInput.includes("cover drive") || cleanInput.includes("drive")) {
        drillKeyword = "cover drive";
    } else if (cleanInput.includes("pull")) {
        drillKeyword = "pull shot";
    } else if (cleanInput.includes("sweep")) {
        drillKeyword = "sweep shot";
    } else if (cleanInput.includes("defensive") || cleanInput.includes("defense") || cleanInput.includes("block")) {
        drillKeyword = "defensive block";
    } else if (cleanInput.includes("fast bowling") || cleanInput.includes("pace") || cleanInput.includes("fast")) {
        drillKeyword = "fast bowling";
    } else if (cleanInput.includes("spin") || cleanInput.includes("spinner")) {
        drillKeyword = "spin bowling";
    }

    if (drillKeyword) {
        const info = COACH_DRILLS[drillKeyword];
        responseText = `
            Excellent question about the <strong>${drillKeyword.toUpperCase()}</strong>. 
            Based on my bio-mechanic diagnostics, here is the optimized training drill for you:<br><br>
            <strong>Drill:</strong> ${info.drillName} (${info.duration})<br>
            <strong>Target:</strong> ${info.target}<br>
            <strong>How-to:</strong> ${info.instructions}<br><br>
            <em>Coach's Pro Tip:</em> ${info.coachTip}
        `;
        
        // Push drill to current active drill board panel
        assignDrillToDashboard(info);
        
    } else if (cleanInput.includes("weakness") || cleanInput.includes("radar") || cleanInput.includes("improve")) {
        // Read active profile role to tailor message
        const role = document.getElementById('profileRoleDisplay').textContent;
        responseText = `
            Looking at your radar diagnostics for your role as a <strong>${role}</strong>, 
            your main biometric discrepancy is <strong>Front Foot Stride length</strong> which is falling 
            around 20% short during cover drives. This causes you to reach for the ball and can trigger edges.
            <br><br>
            I recommend asking me about <strong>"Cover Drive drills"</strong> or checking your Drill Hub to begin 
            correcting this movement pathway.
        `;
    } else if (cleanInput.includes("hello") || cleanInput.includes("hi") || cleanInput.includes("hey")) {
        responseText = `
            Hello Nawin! I am ready. Ask me how to improve specific cricket shots (e.g. <strong>Cover Drive</strong>, <strong>Pull Shot</strong>, <strong>Defensive Block</strong>) 
            or bowling actions (e.g. <strong>Fast Bowling</strong>, <strong>Spin Bowling</strong>) and I will build you a drill sheet.
        `;
    } else {
        responseText = `
            Interesting question, Nawin. I've logged that. 
            To get actionable bio-mechanic feedback, try asking me something like: 
            <em>"How do I improve my cover drive?"</em> or <em>"What drills do you have for fast bowling release height?"</em>.
        `;
    }

    appendBubble('coach', responseText, 'C');
}

// Assigns drill dynamically to the right panel on page
function assignDrillToDashboard(drillInfo) {
    const list = document.getElementById('drillListItems');
    
    // Create new drill element
    const drillBox = document.createElement('div');
    drillBox.className = 'drill-box';
    drillBox.style.borderLeft = `4px solid ${drillInfo.priority === 'CRITICAL' ? 'var(--neon-red)' : 'var(--neon-teal)'}`;
    
    // Prepend to show fresh drill at top
    drillBox.innerHTML = `
        <div class="drill-meta">
            <span class="badge ${drillInfo.priority === 'CRITICAL' ? 'badge-danger' : 'badge-teal'}">${drillInfo.priority}</span>
            <span class="drill-duration"><i class="fa-regular fa-clock"></i> ${drillInfo.duration}</span>
        </div>
        <h4>${drillInfo.drillName}</h4>
        <p>${drillInfo.instructions}</p>
        <div class="drill-target">
            <i class="fa-solid fa-bullseye"></i> Target Benchmark: ${drillInfo.target}
        </div>
    `;
    
    // Remove if duplicate drill name already exists to prevent duplicates
    const existingDrills = list.querySelectorAll('.drill-box h4');
    existingDrills.forEach(el => {
        if (el.textContent === drillInfo.drillName) {
            el.parentElement.remove();
        }
    });
    
    list.prepend(drillBox);
}

// Event Listeners
btnSend.addEventListener('click', sendUserMessage);
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        sendUserMessage();
    }
});

// Helper for typewriter animation inside chat
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .typing-indicator span {
        font-weight: bold;
        font-size: 20px;
        line-height: 8px;
    }
`;
document.head.appendChild(style);
