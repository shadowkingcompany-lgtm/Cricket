/**
 * ai_model.js - Client-Side Neural Network Progress Tracking Model & Visualizer
 * Implements a complete Artificial Neural Network (ANN) in JavaScript
 * and renders a real-time visualization of nodes, weights, and loss curves.
 */

class SimpleNeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        // Initialize weights and biases randomly [-0.5, 0.5]
        this.weightsIH = Array.from({ length: this.hiddenSize }, () => 
            Array.from({ length: this.inputSize }, () => Math.random() - 0.5)
        );
        this.biasH = Array.from({ length: this.hiddenSize }, () => Math.random() - 0.5);

        this.weightsHO = Array.from({ length: this.outputSize }, () => 
            Array.from({ length: this.hiddenSize }, () => Math.random() - 0.5)
        );
        this.biasO = Array.from({ length: this.outputSize }, () => Math.random() - 0.5);
    }

    // Sigmoid Activation
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivative of Sigmoid
    sigmoidDerivative(y) {
        return y * (1 - y);
    }

    // Forward Propagation
    forward(inputs) {
        // Inputs to Hidden
        this.hiddenOutputs = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let sum = this.biasH[i];
            for (let j = 0; j < this.inputSize; j++) {
                sum += inputs[j] * this.weightsIH[i][j];
            }
            this.hiddenOutputs[i] = this.sigmoid(sum);
        }

        // Hidden to Outputs
        this.outputs = [];
        for (let i = 0; i < this.outputSize; i++) {
            let sum = this.biasO[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += this.hiddenOutputs[j] * this.weightsHO[i][j];
            }
            // Linear activation output for regression, scaled through sigmoid for stability
            this.outputs[i] = this.sigmoid(sum);
        }

        return this.outputs;
    }

    // Backward Propagation (Training single step)
    trainStep(inputs, targets, learningRate) {
        // 1. Forward propagate to cache values
        this.forward(inputs);

        // 2. Compute Output Error and Delta
        // Error = Target - Output
        let outputErrors = [];
        let outputDeltas = [];
        for (let i = 0; i < this.outputSize; i++) {
            outputErrors[i] = targets[i] - this.outputs[i];
            // Delta = Error * Sigmoid'
            outputDeltas[i] = outputErrors[i] * this.sigmoidDerivative(this.outputs[i]);
        }

        // 3. Compute Hidden Error and Delta
        let hiddenErrors = [];
        let hiddenDeltas = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let error = 0;
            for (let j = 0; j < this.outputSize; j++) {
                error += outputDeltas[j] * this.weightsHO[j][i];
            }
            hiddenErrors[i] = error;
            hiddenDeltas[i] = hiddenErrors[i] * this.sigmoidDerivative(this.hiddenOutputs[i]);
        }

        // 4. Adjust Weights Hidden -> Output
        for (let i = 0; i < this.outputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                this.weightsHO[i][j] += learningRate * outputDeltas[i] * this.hiddenOutputs[j];
            }
            this.biasO[i] += learningRate * outputDeltas[i];
        }

        // 5. Adjust Weights Input -> Hidden
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.inputSize; j++) {
                this.weightsIH[i][j] += learningRate * hiddenDeltas[i] * inputs[j];
            }
            this.biasH[i] += learningRate * hiddenDeltas[i];
        }

        // Return squared error for tracking
        let sumSquaredError = 0;
        for (let i = 0; i < this.outputSize; i++) {
            sumSquaredError += Math.pow(outputErrors[i], 2);
        }
        return sumSquaredError / this.outputSize;
    }
}

// Visual NN Variables
let nnModel = null;
let nnCanvas = null;
let nnCtx = null;
let lossChart = null;
let trainingInterval = null;
let trainingEpochList = [];
let trainingLossList = [];
let synapsePulseOffset = 0;

// Coordinate layout for visualization nodes
let nodesInput = [];
let nodesHidden = [];
let nodesOutput = [];

// Input Labels for visualizer
const inputLabels = ["Stability", "Stride", "Swing Speed", "Prev Score"];
const outputLabels = ["Progress Index"];

// Initialize canvas and charts
function initNNWorkspace() {
    nnCanvas = document.getElementById('nnCanvas');
    nnCtx = nnCanvas.getContext('2d');
    
    // Fit canvas to element size
    resizeNNCanvas();
    window.addEventListener('resize', resizeNNCanvas);

    // Initialize chart
    const lossCanvas = document.getElementById('lossChart');
    lossChart = new Chart(lossCanvas, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Mean Squared Error (MSE)',
                data: [],
                borderColor: '#ff007f',
                backgroundColor: 'rgba(255, 0, 127, 0.05)',
                borderWidth: 2,
                fill: true,
                tension: 0.2,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Epochs', color: '#94a3b8', font: { family: 'Orbitron' } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Loss', color: '#94a3b8', font: { family: 'Orbitron' } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    // Setup initial random weights visualizer
    const hiddenCount = parseInt(document.getElementById('nnHiddenNodes').value);
    nnModel = new SimpleNeuralNetwork(4, hiddenCount, 1);
    setupNodeLayout(hiddenCount);
    drawNetworkGraph();
    
    // Animation loop for synaptic pulses
    animateSynapses();
}

function resizeNNCanvas() {
    if (!nnCanvas) return;
    const rect = nnCanvas.parentElement.getBoundingClientRect();
    nnCanvas.width = rect.width;
    nnCanvas.height = rect.height - 40; // leaving room for legend
}

function setupNodeLayout(hiddenCount) {
    const w = nnCanvas.width;
    const h = nnCanvas.height;
    
    nodesInput = [];
    nodesHidden = [];
    nodesOutput = [];

    // Inputs: Left column
    const inputX = w * 0.15;
    for (let i = 0; i < 4; i++) {
        nodesInput.push({
            x: inputX,
            y: h * 0.2 + (h * 0.6) * (i / 3),
            label: inputLabels[i]
        });
    }

    // Hidden: Middle column
    const hiddenX = w * 0.5;
    for (let i = 0; i < hiddenCount; i++) {
        nodesHidden.push({
            x: hiddenX,
            y: h * 0.15 + (h * 0.7) * (i / (hiddenCount - 1))
        });
    }

    // Output: Right column
    const outputX = w * 0.85;
    nodesOutput.push({
        x: outputX,
        y: h * 0.5,
        label: outputLabels[0]
    });
}

// Main paint loop for network graph
function drawNetworkGraph() {
    if (!nnCtx) return;
    nnCtx.clearRect(0, 0, nnCanvas.width, nnCanvas.height);

    // 1. Draw Synapses (Weights)
    // Input to Hidden
    for (let h = 0; h < nnModel.hiddenSize; h++) {
        for (let i = 0; i < nnModel.inputSize; i++) {
            const wVal = nnModel.weightsIH[h][i];
            drawSynapse(nodesInput[i], nodesHidden[h], wVal);
        }
    }

    // Hidden to Output
    for (let o = 0; o < nnModel.outputSize; o++) {
        for (let h = 0; h < nnModel.hiddenSize; h++) {
            const wVal = nnModel.weightsHO[o][h];
            drawSynapse(nodesHidden[h], nodesOutput[o], wVal);
        }
    }

    // 2. Draw Synaptic Signal Pulses (Forward passes visual)
    if (trainingInterval) {
        synapsePulseOffset += 0.05;
        if (synapsePulseOffset > 1.0) synapsePulseOffset = 0;

        // Input to Hidden pulses
        for (let h = 0; h < nnModel.hiddenSize; h++) {
            for (let i = 0; i < nnModel.inputSize; i++) {
                drawPulse(nodesInput[i], nodesHidden[h], synapsePulseOffset);
            }
        }
        
        // Hidden to Output pulses
        for (let o = 0; o < nnModel.outputSize; o++) {
            for (let h = 0; h < nnModel.hiddenSize; h++) {
                drawPulse(nodesHidden[h], nodesOutput[o], synapsePulseOffset);
            }
        }
    }

    // 3. Draw Nodes
    // Draw Inputs
    nodesInput.forEach(node => {
        drawNode(node.x, node.y, 'rgba(0, 242, 254, 0.2)', 'var(--neon-teal)');
        // Label
        nnCtx.fillStyle = '#94a3b8';
        nnCtx.font = '10px Orbitron';
        nnCtx.textAlign = 'right';
        nnCtx.fillText(node.label, node.x - 18, node.y + 4);
    });

    // Draw Hidden
    nodesHidden.forEach(node => {
        drawNode(node.x, node.y, 'rgba(255, 0, 127, 0.15)', 'var(--neon-magenta)');
    });

    // Draw Output
    nodesOutput.forEach(node => {
        drawNode(node.x, node.y, 'rgba(57, 255, 20, 0.2)', 'var(--neon-green)');
        // Label
        nnCtx.fillStyle = '#ffffff';
        nnCtx.font = 'bold 11px Orbitron';
        nnCtx.textAlign = 'left';
        nnCtx.fillText(node.label, node.x + 18, node.y + 4);
    });
}

// Synapse line drawing helper
function drawSynapse(p1, p2, weight) {
    nnCtx.beginPath();
    nnCtx.moveTo(p1.x, p1.y);
    nnCtx.lineTo(p2.x, p2.y);
    
    // Weight value decides color: blue for positive, red for negative
    nnCtx.strokeStyle = weight >= 0 ? 'rgba(59, 130, 246, 0.4)' : 'rgba(239, 68, 68, 0.4)';
    
    // Weight decides thickness
    nnCtx.lineWidth = Math.min(8, 0.5 + Math.abs(weight) * 6);
    nnCtx.stroke();
}

// Moving pulse helper
function drawPulse(p1, p2, offset) {
    const pulseX = p1.x + (p2.x - p1.x) * offset;
    const pulseY = p1.y + (p2.y - p1.y) * offset;
    
    nnCtx.beginPath();
    nnCtx.arc(pulseX, pulseY, 3, 0, 2 * Math.PI);
    nnCtx.fillStyle = 'var(--neon-magenta)';
    nnCtx.shadowBlur = 8;
    nnCtx.shadowColor = 'var(--neon-magenta)';
    nnCtx.fill();
    nnCtx.shadowBlur = 0; // reset
}

// Node circle drawing helper
function drawNode(x, y, bg, border) {
    nnCtx.beginPath();
    nnCtx.arc(x, y, 12, 0, 2 * Math.PI);
    nnCtx.fillStyle = bg;
    nnCtx.fill();
    
    nnCtx.beginPath();
    nnCtx.arc(x, y, 12, 0, 2 * Math.PI);
    nnCtx.strokeStyle = border;
    nnCtx.lineWidth = 2.5;
    nnCtx.stroke();
}

// Animation loop
function animateSynapses() {
    drawNetworkGraph();
    requestAnimationFrame(animateSynapses);
}

// Train Neural Network Model Asynchronously
function startTrainingNN() {
    if (trainingInterval) return;

    // Reset controls
    document.getElementById('btnTrainNN').disabled = true;
    document.getElementById('btnQueryNN').disabled = true;
    document.getElementById('nnStatusBadge').textContent = "TRAINING...";
    document.getElementById('nnStatusBadge').className = "badge badge-magenta";

    const epochsTarget = parseInt(document.getElementById('nnEpochs').value);
    const learningRate = parseFloat(document.getElementById('nnLR').value);
    const hiddenCount = parseInt(document.getElementById('nnHiddenNodes').value);

    // Reinitialize NN model structure
    nnModel = new SimpleNeuralNetwork(4, hiddenCount, 1);
    setupNodeLayout(hiddenCount);

    // Gather or synthesize session datasets to train on
    // Normalized inputs: [Stability, Stride, Speed, Prev Score] -> Target Progress Rating (0-1)
    let trainingData = [];
    
    // Check if we have logs saved in the UI session database table
    const tableRows = document.querySelectorAll('#sessionTable tbody tr');
    if (tableRows.length > 3 && !tableRows[0].classList.contains('empty-item')) {
        // Build dataset from actual user practice sessions
        tableRows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length > 7) {
                // Parse and normalize
                const isBatting = cells[2].textContent.toLowerCase() === 'batting';
                const stabilityStr = cells[4].textContent;
                const stability = parseFloat(stabilityStr) / 100.0;
                
                const metric2Str = cells[5].textContent;
                const strideOrHt = parseFloat(metric2Str);
                const normalizedStride = isBatting ? strideOrHt / 1.6 : strideOrHt / 2.5;
                
                const speedStr = cells[6].textContent;
                const speed = parseFloat(speedStr);
                const normalizedSpeed = speed / 850.0;
                
                const score = parseFloat(cells[7].textContent) / 100.0;
                
                // Target index computed
                const targetRating = 0.3 * stability + 0.2 * normalizedStride + 0.3 * normalizedSpeed + 0.2 * score;
                trainingData.push({
                    input: [stability, normalizedStride, normalizedSpeed, score],
                    output: [targetRating]
                });
            }
        });
    }
    
    // Supplement/Create synthetic sports science correlation dataset so it fits a solid progression function
    // High stability, stride, and speed ALWAYS leads to high output rating (ANN learns this function)
    const baseSyntheticSize = 35;
    for (let k = 0; k < baseSyntheticSize; k++) {
        const stab = 0.5 + Math.random() * 0.5; // [0.5, 1.0]
        const stride = 0.4 + Math.random() * 0.6; // [0.4, 1.0]
        const speed = 0.3 + Math.random() * 0.7; // [0.3, 1.0]
        const prev = 0.5 + Math.random() * 0.5; // [0.5, 1.0]
        
        // Progression Math formula (target score)
        const rating = 0.35 * stab + 0.2 * stride + 0.3 * speed + 0.15 * prev;
        trainingData.push({
            input: [stab, stride, speed, prev],
            output: [rating]
        });
    }

    let epoch = 0;
    trainingEpochList = [];
    trainingLossList = [];

    // Async loop to visualize progress epoch by epoch without blocking UI thread
    trainingInterval = setInterval(() => {
        let totalLoss = 0;
        
        // Train on all samples (Epoch step)
        trainingData.forEach(sample => {
            const loss = nnModel.trainStep(sample.input, sample.output, learningRate);
            totalLoss += loss;
        });

        const avgLoss = totalLoss / trainingData.length;
        epoch++;

        // Update UI displays
        document.getElementById('nnEpochDisplay').textContent = epoch;
        document.getElementById('nnLossDisplay').textContent = avgLoss.toFixed(5);

        // Record loss every epoch for chart (cap chart size to avoid performance lag)
        if (epoch % Math.max(1, Math.floor(epochsTarget / 50)) === 0 || epoch === epochsTarget) {
            trainingEpochList.push(epoch);
            trainingLossList.push(avgLoss);
            
            lossChart.data.labels = trainingEpochList;
            lossChart.data.datasets[0].data = trainingLossList;
            lossChart.update('none'); // Update quickly without animations
        }

        if (epoch >= epochsTarget) {
            // Training finished!
            clearInterval(trainingInterval);
            trainingInterval = null;
            
            document.getElementById('btnTrainNN').disabled = false;
            document.getElementById('btnQueryNN').disabled = false;
            document.getElementById('nnStatusBadge').textContent = "CONVERGED";
            document.getElementById('nnStatusBadge').className = "badge badge-green";
            document.getElementById('nnWiringStatus').textContent = "MODEL OPTIMIZED";
            document.getElementById('nnWiringStatus').className = "badge badge-green";
            
            // Auto trigger query output
            queryTrainedNN();
        }
    }, 12); // ~80 Epochs/second speed
}

// Reset weights randomly
function resetNNWeights() {
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
    
    const hiddenCount = parseInt(document.getElementById('nnHiddenNodes').value);
    nnModel = new SimpleNeuralNetwork(4, hiddenCount, 1);
    setupNodeLayout(hiddenCount);

    document.getElementById('nnEpochDisplay').textContent = "0";
    document.getElementById('nnLossDisplay').textContent = "0.0000";
    document.getElementById('nnStatusBadge').textContent = "UNTRAINED";
    document.getElementById('nnStatusBadge').className = "badge badge-outline";
    document.getElementById('nnWiringStatus').textContent = "WIRING ACTIVE";
    document.getElementById('nnWiringStatus').className = "badge badge-magenta animate-pulse";
    document.getElementById('btnTrainNN').disabled = false;
    document.getElementById('btnQueryNN').disabled = true;

    // Reset loss chart
    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];
    lossChart.update();

    // Reset Predictor Box
    document.getElementById('nnPredictedIndex').textContent = "--";
    document.getElementById('nnPredictedText').textContent = "Model not trained. Complete training to run predictions.";
}

// Forward pass query custom parameters in tester card
function queryTrainedNN() {
    if (!nnModel) return;

    // Get input telemetry
    const stability = parseFloat(document.getElementById('nnTestStability').value);
    const stride = parseFloat(document.getElementById('nnTestStride').value);
    const speed = parseFloat(document.getElementById('nnTestSpeed').value);
    const prevScore = parseFloat(document.getElementById('nnTestScore').value);

    // Normalize values
    const nStability = stability / 100.0;
    const nStride = Math.min(1.0, stride / 1.6);
    const nSpeed = Math.min(1.0, speed / 850.0);
    const nPrev = prevScore / 100.0;

    // Forward pass
    const outputs = nnModel.forward([nStability, nStride, nSpeed, nPrev]);
    // De-normalize output rating (Sigmoid maps 0-1 to continuous rating index 0-100)
    const ratingIndex = Math.round(outputs[0] * 100);

    // Update screen
    const valueEl = document.getElementById('nnPredictedIndex');
    const textEl = document.getElementById('nnPredictedText');

    valueEl.textContent = ratingIndex;

    // Diagnose verbal progress level
    if (ratingIndex >= 85) {
        textEl.textContent = "AI DIAGNOSIS: Elite Tier Progress. Stance consistency & arm kinetic output are exceptionally aligned.";
        valueEl.style.color = "var(--neon-green)";
    } else if (ratingIndex >= 70) {
        textEl.textContent = "AI DIAGNOSIS: Strong Competitive Level. Stabilized form. Stride extension could increase to fully optimize power.";
        valueEl.style.color = "var(--neon-teal)";
    } else if (ratingIndex >= 55) {
        textEl.textContent = "AI DIAGNOSIS: Steady Intermediate. Noticeable kinetic lag in arm swing. Drill center recommendations suggested.";
        valueEl.style.color = "#eab308";
    } else {
        textEl.textContent = "AI DIAGNOSIS: Development Required. Joint stability fluctuates. Prioritize basic alignment drills.";
        valueEl.style.color = "var(--neon-red)";
    }
}

// Attach Event Listeners
document.getElementById('btnTrainNN').addEventListener('click', startTrainingNN);
document.getElementById('btnResetNN').addEventListener('click', resetNNWeights);
document.getElementById('btnQueryNN').addEventListener('click', queryTrainedNN);

document.getElementById('nnHiddenNodes').addEventListener('change', () => {
    resetNNWeights();
});

// Initialize workspace when DOM loaded
window.addEventListener('DOMContentLoaded', () => {
    initNNWorkspace();
});
