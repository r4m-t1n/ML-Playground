document.addEventListener('DOMContentLoaded', () => {
    const tabButtons = document.querySelectorAll('.tab-button');
    const modelContentArea = document.getElementById('model-content');

    const modelStates = {
        'simple-linear-regression': { chart: null, data: [], defaultData: '1,2.5;2,4.1;3,5.8' },
        'simple-classifier': { chart: null, data: { class0: [], class1: [] }, defaultData: '1,2,0;3,4,1;5,1,0;6,3,1' },
        'mlp-regression': { chart: null, data: [], defaultData: '1,1;2,3;3,2;4,4;5,3;6,5;7,4;8,6;9,5' }
    };

    const modelTemplates = {
        'simple-linear-regression': `
            <h2>Linear Regression Model</h2>
            <div class="model-section">
                <div class="chart-container">
                    <canvas id="chart-simple-linear-regression"></canvas>
                </div>
                <div class="controls-container">
                    <h3>Parameters & Data Input</h3>
                    <div class="control-group data-input-group">
                        <label for="input-data-simple-linear-regression">Enter Data Points (e.g., x1,y1;x2,y2):</label>
                        <textarea id="input-data-simple-linear-regression" rows="5" placeholder="Example: 1,2.5;2,4.1;3,5.8"></textarea>
                        <p>Separate points with semicolons (;), and x,y values with commas (,).</p>
                    </div>

                    <div class="control-group">
                        <label for="slider-lr-learning-rate">Learning Rate:</label>
                        <div class="slider-container">
                            <input type="range" id="slider-lr-learning-rate" min="0.0001" max="0.1" step="0.0001" value="0.01">
                            <span id="value-lr-learning-rate">0.01</span>
                        </div>
                    </div>
                    <div class="control-group">
                        <label for="slider-lr-epochs">Epochs:</label>
                        <div class="slider-container">
                            <input type="range" id="slider-lr-epochs" min="10" max="1000" step="10" value="100">
                            <span id="value-lr-epochs">100</span>
                        </div>
                    </div>
                    <div class="button-group">
                        <button id="button-train-simple-linear-regression">Train Model</button>
                        <button id="button-reset-simple-linear-regression">Reset Data & Model</button>
                    </div>
                    <div class="results-area">
                        <h4>Training Loss: <span id="display-lr-loss">N/A</span></h4>
                        <p>Model Weights (w, b): <span id="display-lr-weights">N/A</span></p>
                    </div>
                </div>
            </div>
        `,
        'simple-classifier': `
            <h2>Simple Classifier Model</h2>
            <div class="model-section">
                <div class="chart-container">
                    <canvas id="chart-simple-classifier"></canvas>
                </div>
                <div class="controls-container">
                    <h3>Parameters & Data Input</h3>
                    <div class="control-group data-input-group">
                        <label for="input-data-simple-classifier">Enter Data Points (e.g., x1,y1,label1;x2,y2,label2):</label>
                        <textarea id="input-data-simple-classifier" rows="5" placeholder="Example: 1,2,0;3,4,1;5,1,0"></textarea>
                        <p>Separate points with semicolons (;), and x,y,label values with commas (,). Label 0 or 1.</p>
                    </div>

                    <div class="control-group">
                        <label for="slider-cls-learning-rate">Learning Rate:</label>
                        <div class="slider-container">
                            <input type="range" id="slider-cls-learning-rate" min="0.0001" max="0.1" step="0.0001" value="0.01">
                            <span id="value-cls-learning-rate">0.01</span>
                        </div>
                    </div>
                    <div class="control-group">
                        <label for="slider-cls-epochs">Epochs:</label>
                        <div class="slider-container">
                            <input type="range" id="slider-cls-epochs" min="10" max="1000" step="10" value="100">
                            <span id="value-cls-epochs">100</span>
                        </div>
                    </div>
                    <div class="button-group">
                        <button id="button-train-simple-classifier">Train Model</button>
                        <button id="button-reset-simple-classifier">Reset Data & Model</button>
                    </div>
                    <div class="results-area">
                        <h4>Training Loss: <span id="display-cls-loss">N/A</span></h4>
                        <p>Decision Boundary Info: <span id="display-cls-boundary">N/A</span></p>
                    </div>
                </div>
            </div>
        `,
        'mlp-regression': `
            <h2>MLP Regression Model</h2>
            <div class="model-section">
                <div class="chart-container">
                    <canvas id="chart-mlp-regression"></canvas>
                </div>
                <div class="controls-container">
                    <h3>Parameters & Data Input</h3>
                    <div class="control-group data-input-group">
                        <label for="input-data-mlp-regression">Enter Data Points (e.g., x1,y1;x2,y2):</label>
                        <textarea id="input-data-mlp-regression" rows="5" placeholder="Example: 1,2.5;2,4.1;3,5.8"></textarea>
                        <p>Separate points with semicolons (;), and x,y values with commas (,).</p>
                    </div>

                    <div class="control-group">
                        <label for="slider-mlp-learning-rate">Learning Rate:</label>
                        <div class="slider-container">
                            <input type="range" id="slider-mlp-learning-rate" min="0.0001" max="0.1" step="0.0001" value="0.01">
                            <span id="value-mlp-learning-rate">0.01</span>
                        </div>
                    </div>
                    <div class="control-group">
                        <label for="slider-mlp-epochs">Epochs:</label>
                        <div class="slider-container">
                            <input type="range" id="slider-mlp-epochs" min="10" max="1000" step="10" value="100">
                            <span id="value-mlp-epochs">100</span>
                        </div>
                    </div>
                    <div class="control-group">
                        <label for="slider-mlp-hidden-size">Hidden Layer Size:</label>
                        <div class="slider-container">
                            <input type="range" id="slider-mlp-hidden-size" min="5" max="50" step="5" value="20">
                            <span id="value-mlp-hidden-size">20</span>
                        </div>
                    </div>
                    <div class="button-group">
                        <button id="button-train-mlp-regression">Train Model</button>
                        <button id="button-reset-mlp-regression">Reset Data & Model</button>
                    </div>
                     <div class="results-area">
                        <h4>Training Loss: <span id="display-mlp-loss">N/A</span></h4>
                    </div>
                </div>
            </div>
        `
    };

    function setupSliderValueDisplay(modelName, sliderIdSuffix) {
        const slider = document.getElementById(`slider-${modelName}-${sliderIdSuffix}`);
        const valueSpan = document.getElementById(`value-${modelName}-${sliderIdSuffix}`);
        if (slider && valueSpan) {
            valueSpan.textContent = slider.value;
            slider.addEventListener('input', () => {
                valueSpan.textContent = slider.value;
            });
        }
    }

    function renderChart(canvasId, chartType, data, options, modelName) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        if (modelStates[modelName].chart) {
            modelStates[modelName].chart.destroy();
        }
        modelStates[modelName].chart = new Chart(ctx, {
            type: chartType,
            data: data,
            options: options
        });
        return modelStates[modelName].chart;
    }

    function parseDataInput(inputString, type = 'regression') {
        const points = inputString.split(';').map(s => s.trim()).filter(s => s);
        if (type === 'regression') {
            return points.map(p => {
                const [x, y] = p.split(',').map(Number);
                if (isNaN(x) || isNaN(y)) throw new Error("Invalid data format. Use x,y");
                return { x, y };
            });
        } else if (type === 'classifier') {
            return points.map(p => {
                const [x, y, label] = p.split(',').map(Number);
                if (isNaN(x) || isNaN(y) || isNaN(label) || (label !== 0 && label !== 1)) {
                    throw new Error("Invalid data format. Use x,y,label (label must be 0 or 1)");
                }
                return { x, y, label };
            });
        }
        return [];
    }

    async function handleTrain(modelName) {
        const dataInputElem = document.getElementById(`input-data-${modelName}`);
        let currentDataPoints;
        try {
            if (modelName === 'simple-classifier') {
                const parsed = parseDataInput(dataInputElem.value, 'classifier');
                modelStates[modelName].data.class0 = parsed.filter(p => p.label === 0);
                modelStates[modelName].data.class1 = parsed.filter(p => p.label === 1);
                currentDataPoints = parsed;
            } else {
                currentDataPoints = parseDataInput(dataInputElem.value, 'regression');
            }
        } catch (e) {
            alert(e.message);
            return;
        }

        if (currentDataPoints.length === 0) {
            alert("Please enter some data points first!");
            return;
        }

        const learningRate = parseFloat(document.getElementById(`slider-${modelName}-learning-rate`).value);
        const epochs = parseInt(document.getElementById(`slider-${modelName}-epochs`).value);
        
        let requestBody = {
            data: currentDataPoints,
            learning_rate: learningRate,
            epochs: epochs
        };

        if (modelName === 'mlp-regression') {
            requestBody.hidden_size = parseInt(document.getElementById(`slider-${modelName}-hidden-size`).value);
        }

        try {
            const response = await fetch(`/train_${modelName.replace('-', '_')}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });
            const result = await response.json();
            
            document.getElementById(`display-${modelName}-loss`).textContent = result.final_loss.toFixed(6);

            const chart = modelStates[modelName].chart;
            if (!chart) return;

            if (modelName === 'simple-classifier') {
                chart.data.datasets[0].data = modelStates[modelName].data.class0;
                chart.data.datasets[1].data = modelStates[modelName].data.class1;
            } else {
                chart.data.datasets[0].data = currentDataPoints;
            }

            if (modelName === 'simple-linear-regression') {
                const w = result.weights[0];
                const b = result.bias;
                document.getElementById(`display-lr-weights`).textContent = `w=${w.toFixed(4)}, b=${b.toFixed(4)}`;
                const lineData = [{ x: 0, y: b }, { x: 10, y: w * 10 + b }];
                chart.data.datasets[1].data = lineData;
            } else if (modelName === 'simple-classifier') {
                const w = result.weights;
                const b = result.bias;
                let boundaryData = [];
                let boundaryEquation = "N/A";
                const maxX = chart.options.scales.x.max; 
                const maxY = chart.options.scales.y.max; 

                if (w && w.length === 2 && b !== undefined) {
                    if (w[1] !== 0) {
                        const m = -w[0] / w[1];
                        const c = -b / w[1];
                        boundaryData.push({ x: 0, y: c });
                        boundaryData.push({ x: maxX, y: m * maxX + c });
                        boundaryEquation = `y = ${m.toFixed(4)}x + ${c.toFixed(4)}`;
                    } else if (w[0] !== 0) {
                        const c = -b / w[0];
                        boundaryData.push({ x: c, y: 0 });
                        boundaryData.push({ x: c, y: maxY });
                        boundaryEquation = `x = ${c.toFixed(4)}`;
                    } else {
                        boundaryEquation = "No clear boundary (weights are zero)";
                    }
                }
                document.getElementById(`display-cls-boundary`).textContent = boundaryEquation;
                chart.data.datasets[2].data = boundaryData;

            } else if (modelName === 'mlp-regression') {
                const predictionInputs = Array.from({ length: 101 }, (_, i) => i * (chart.options.scales.x.max / 100));
                const predictResponse = await fetch('/predict_mlp-regression', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ inputs: predictionInputs })
                });
                const predictResult = await predictResponse.json();

                const mlpLineData = predictionInputs.map((x, index) => ({
                    x: x,
                    y: predictResult.predictions[index]
                }));
                chart.data.datasets[1].data = mlpLineData;
            }
            chart.update();

        } catch (error) {
            console.error(`Error training ${modelName} model:`, error);
            document.getElementById(`display-${modelName}-loss`).textContent = 'Error';
            if (modelName === 'simple-linear-regression') document.getElementById(`display-lr-weights`).textContent = 'Error';
            if (modelName === 'simple-classifier') document.getElementById(`display-cls-boundary`).textContent = 'Error';
        }
    }

    function handleReset(modelName) {
        const dataInputElem = document.getElementById(`input-data-${modelName}`);
        dataInputElem.value = modelStates[modelName].defaultData;

        if (modelName === 'simple-classifier') {
            const parsed = parseDataInput(dataInputElem.value, 'classifier');
            modelStates[modelName].data.class0 = parsed.filter(p => p.label === 0);
            modelStates[modelName].data.class1 = parsed.filter(p => p.label === 1);
        } else {
            modelStates[modelName].data = parseDataInput(dataInputElem.value, 'regression');
        }

        const chart = modelStates[modelName].chart;
        if (chart) {
             if (modelName === 'simple-classifier') {
                chart.data.datasets[0].data = modelStates[modelName].data.class0;
                chart.data.datasets[1].data = modelStates[modelName].data.class1;
                chart.data.datasets[2].data = [];
            } else {
                chart.data.datasets[0].data = modelStates[modelName].data;
                chart.data.datasets[1].data = [];
            }
            chart.update();
        }
        document.getElementById(`display-${modelName}-loss`).textContent = 'N/A';
        if (modelName === 'simple-linear-regression') document.getElementById(`display-lr-weights`).textContent = 'N/A';
        if (modelName === 'simple-classifier') document.getElementById(`display-cls-boundary`).textContent = 'N/A';
    }

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const modelName = button.dataset.model;
            activateTab(modelName);
        });
    });

    function activateTab(modelName) {
        tabButtons.forEach(button => button.classList.remove('active'));
        const activeTabButton = document.querySelector(`.tab-button[data-model="${modelName}"]`);
        if (activeTabButton) {
            activeTabButton.classList.add('active');
        }

        modelContentArea.innerHTML = modelTemplates[modelName] || '<h2>Model not found.</h2>';

        setupSliderValueDisplay(modelName, 'learning-rate');
        setupSliderValueDisplay(modelName, 'epochs');
        if (modelName === 'mlp-regression') {
            setupSliderValueDisplay(modelName, 'hidden-size');
        }

        const dataInputElem = document.getElementById(`input-data-${modelName}`);
        dataInputElem.value = modelStates[modelName].defaultData;

        let chartData;
        let chartOptions;

        if (modelName === 'simple-linear-regression') {
            const parsedData = parseDataInput(dataInputElem.value, 'regression');
            modelStates[modelName].data = parsedData;
            chartData = {
                datasets: [
                    { label: 'Data Points', data: parsedData, backgroundColor: 'rgba(75, 192, 192, 0.6)', pointRadius: 6 },
                    { label: 'Regression Line', data: [], borderColor: 'red', type: 'line', fill: false, pointRadius: 0, borderWidth: 2 }
                ]
            };
            chartOptions = { scales: { x: { type: 'linear', position: 'bottom', min: 0, max: 10 }, y: { type: 'linear', position: 'left', min: 0, max: 10 } } };
        } else if (modelName === 'simple-classifier') {
            const parsed = parseDataInput(dataInputElem.value, 'classifier');
            modelStates[modelName].data.class0 = parsed.filter(p => p.label === 0);
            modelStates[modelName].data.class1 = parsed.filter(p => p.label === 1);
            chartData = {
                datasets: [
                    { label: 'Class 0', data: modelStates[modelName].data.class0, backgroundColor: 'red', pointRadius: 6 },
                    { label: 'Class 1', data: modelStates[modelName].data.class1, backgroundColor: 'green', pointRadius: 6 },
                    { label: 'Decision Boundary', data: [], borderColor: 'purple', type: 'line', fill: false, pointRadius: 0, borderWidth: 2 }
                ]
            };
            chartOptions = { scales: { x: { type: 'linear', position: 'bottom', min: 0, max: 10 }, y: { type: 'linear', position: 'left', min: 0, max: 10 } } };
        } else if (modelName === 'mlp-regression') {
            const parsedData = parseDataInput(dataInputElem.value, 'regression');
            modelStates[modelName].data = parsedData;
             chartData = {
                datasets: [
                    { label: 'Data Points', data: parsedData, backgroundColor: 'rgba(75, 192, 192, 0.6)', pointRadius: 6 },
                    { label: 'MLP Prediction', data: [], borderColor: 'blue', type: 'line', fill: false, pointRadius: 0, borderWidth: 2 }
                ]
            };
            chartOptions = { scales: { x: { type: 'linear', position: 'bottom', min: 0, max: 10 }, y: { type: 'linear', position: 'left', min: 0, max: 10 } } };
        }

        renderChart(`chart-${modelName}`, 'scatter', chartData, chartOptions, modelName);

        const trainButton = document.getElementById(`button-train-${modelName}`);
        if (trainButton) trainButton.addEventListener('click', () => handleTrain(modelName));

        const resetButton = document.getElementById(`button-reset-${modelName}`);
        if (resetButton) resetButton.addEventListener('click', () => handleReset(modelName));

        dataInputElem.addEventListener('input', () => {
            try {
                if (modelName === 'simple-classifier') {
                    const parsed = parseDataInput(dataInputElem.value, 'classifier');
                    modelStates[modelName].data.class0 = parsed.filter(p => p.label === 0);
                    modelStates[modelName].data.class1 = parsed.filter(p => p.label === 1);
                    modelStates[modelName].chart.data.datasets[0].data = modelStates[modelName].data.class0;
                    modelStates[modelName].chart.data.datasets[1].data = modelStates[modelName].data.class1;
                } else {
                    modelStates[modelName].data = parseDataInput(dataInputElem.value, 'regression');
                    modelStates[modelName].chart.data.datasets[0].data = modelStates[modelName].data;
                }
                if (modelStates[modelName].chart.data.datasets[1]) {
                    modelStates[modelName].chart.data.datasets[1].data = [];
                }
                if (modelStates[modelName].chart.data.datasets[2]) {
                     modelStates[modelName].chart.data.datasets[2].data = [];
                }
                modelStates[modelName].chart.update();
            } catch (e) {
                console.warn(`Invalid data format for ${modelName}:`, e.message);
            }
        });

        document.getElementById(`display-${modelName}-loss`).textContent = 'N/A';
        if (modelName === 'simple-linear-regression') document.getElementById(`display-lr-weights`).textContent = 'N/A';
        if (modelName === 'simple-classifier') document.getElementById(`display-cls-boundary`).textContent = 'N/A';
    }

    if (tabButtons.length > 0) {
        activateTab(tabButtons[0].dataset.model);
    }
});