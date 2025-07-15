document.addEventListener('DOMContentLoaded', () => {
    const tabButtons = document.querySelectorAll('.tab-button');
    const modelContentArea = document.getElementById('model-content');

    const modelStates = {
        'simple-linear-regression': { chart: null, data: [], defaultData: '1,2.5;2,4.1;3,5.8' },
        'simple-classifier': { chart: null, data: { class0: [], class1: [] }, defaultData: '1,2,0;3,4,1;5,1,0;6,3,1' },
        'mlp-regression': { chart: null, data: [], defaultData: '1,1;2,3;3,2;4,4;5,3;6,5;7,4;8,6;9,5' },
        'mlp-classifier': { chart: null, data: { class0: [], class1: [] }, defaultData: '1,2,0;3,4,1;5,1,0;6,3,1' }
    };

    const modelTemplates = {
        'simple-linear-regression': 'simple-linear-regression.html',
        'simple-classifier': 'simple-classifier.html',
        'mlp-regression': 'mlp-regression.html',
        'mlp-classifier': 'mlp-classifier.html'
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
        let requestBody;
        try {
            if (modelName === 'simple-classifier' || modelName === 'mlp-classifier') {
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
        requestBody = {
            data: currentDataPoints,
            learning_rate: learningRate,
            epochs: epochs
        };

        if (modelName === 'mlp-regression' || modelName === 'mlp-classifier') {
            requestBody.hidden_size = parseInt(document.getElementById(`slider-${modelName}-hidden-size`).value);
        }
        
        try {
            const backendTrainEndpoint = `/train_${modelName.replace(/-/g, '_')}`; 
            const response = await fetch(backendTrainEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });
            const result = await response.json();
            
            if (response.status !== 200) {
                throw new Error(result.detail || `Backend error: Status ${response.status}`);
            }

            document.getElementById(`display-${modelName}-loss`).textContent = result.final_loss.toFixed(6);

            const chart = modelStates[modelName].chart;
            if (!chart) return;

            if (modelName === 'simple-classifier') {
                chart.data.datasets[0].data = modelStates[modelName].data.class0;
                chart.data.datasets[1].data = modelStates[modelName].data.class1;
            } else {
                chart.data.datasets[0].data = currentDataPoints;
            }

            if (modelName === 'mlp-classifier') {
                const chart = modelStates[modelName].chart;
                if (!chart) return;

                const parsed = parseDataInput(dataInputElem.value, 'classifier');
                modelStates[modelName].data.class0 = parsed.filter(p => p.label === 0);
                modelStates[modelName].data.class1 = parsed.filter(p => p.label === 1);
                chart.data.datasets[0].data = modelStates[modelName].data.class0;
                chart.data.datasets[1].data = modelStates[modelName].data.class1;

                const gridPoints = [];
                const xSteps = 50;
                const ySteps = 50;
                const xMin = chart.options.scales.x.min;
                const xMax = chart.options.scales.x.max;
                const yMin = chart.options.scales.y.min;
                const yMax = chart.options.scales.y.max;
                
                for (let i = 0; i <= xSteps; i++) {
                    for (let j = 0; j <= ySteps; j++) {
                        const x = xMin + (i / xSteps) * (xMax - xMin);
                        const y = yMin + (j / ySteps) * (yMax - yMin);
                        gridPoints.push([x, y]);
                    }
                }

                const predictResponse = await fetch('/predict_mlp_classifier', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ inputs: gridPoints })
                });
                const predictResult = await predictResponse.json();

                if (predictResponse.status !== 200) {
                    throw new Error(predictResult.detail || "Unknown prediction error from backend.");
                }

                const boundaryData = predictResult.predictions.map((p, index) => ({
                    x: gridPoints[index][0],
                    y: gridPoints[index][1],
                    color: p === 1 ? 'rgba(0, 128, 0, 0.1)' : 'rgba(255, 0, 0, 0.1)'
                }));
                chart.data.datasets[2].data = boundaryData.filter(d => d.color === 'rgba(255, 0, 0, 0.1)');
                chart.data.datasets[3].data = boundaryData.filter(d => d.color === 'rgba(0, 128, 0, 0.1)');
                
                chart.update();

            } else if (modelName === 'simple-linear-regression') {
                const w = result.weights[0];
                const b = result.bias;
                document.getElementById(`display-simple-linear-regression-weights`).textContent = `w=${w.toFixed(4)}, b=${b.toFixed(4)}`;
                
                const maxX = chart.options.scales.x.max;
                const lineData = [{ x: 0, y: b }, { x: maxX, y: w * maxX + b }];
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
                document.getElementById(`display-simple-classifier-boundary`).textContent = boundaryEquation;
                chart.data.datasets[2].data = boundaryData;

            } else if (modelName === 'mlp-regression') {
                const predictionInputs = Array.from({ length: 101 }, (_, i) => i * (chart.options.scales.x.max / 100));
                const predictResponse = await fetch('/predict_mlp_regression', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ inputs: predictionInputs })
                });
                const predictResult = await predictResponse.json();
                if (predictResponse.status !== 200) {
                    throw new Error(predictResult.detail || "Unknown prediction error from backend.");
                }

                const mlpLineData = predictionInputs.map((x, index) => ({
                    x: x,
                    y: predictResult.predictions[index]
                }));
                chart.data.datasets[1].data = mlpLineData;
            }
            chart.update();

        } catch (error) {
            console.error(`Error training ${modelName} model:`, error);
            alert(`Error during training: ${error.message}. Check console for details.`);
            document.getElementById(`display-${modelName}-loss`).textContent = 'Error';
            if (document.getElementById(`display-${modelName}-weights`)) {
                document.getElementById(`display-${modelName}-weights`).textContent = 'Error';
            }
            if (document.getElementById(`display-${modelName}-boundary`)) {
                document.getElementById(`display-${modelName}-boundary`).textContent = 'Error';
            }
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
        if (document.getElementById(`display-${modelName}-weights`)) {
            document.getElementById(`display-${modelName}-weights`).textContent = 'N/A';
        }
        if (document.getElementById(`display-${modelName}-boundary`)) {
            document.getElementById(`display-${modelName}-boundary`).textContent = 'N/A';
        }
    }

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const modelName = button.dataset.model;
            activateTab(modelName);
        });
    });

    async function activateTab(modelName) {
        tabButtons.forEach(button => button.classList.remove('active'));
        const activeTabButton = document.querySelector(`.tab-button[data-model="${modelName}"]`);
        if (activeTabButton) {
            activeTabButton.classList.add('active');
        }

        const templateFile = modelTemplates[modelName];
        if (!templateFile) {
            modelContentArea.innerHTML = '<h2>Model not found.</h2>';
            return;
        }

        try {
            const response = await fetch(`/templates/${templateFile}`);
            if (!response.ok) {
                throw new Error(`Failed to load ${templateFile}. Status: ${response.status}`);
            }
            const htmlContent = await response.text();
            modelContentArea.innerHTML = htmlContent;
            setTimeout(() => {
                
                setupSliderValueDisplay(modelName, 'learning-rate');
                setupSliderValueDisplay(modelName, 'epochs');
                if (modelName === 'mlp-regression' || modelName === 'mlp-classifier' || modelName === 'kmeans') {
                    setupSliderValueDisplay(modelName, 'hidden-size');
                }

                const dataInputElem = document.getElementById(`input-data-${modelName}`);
                dataInputElem.value = modelStates[modelName].defaultData;

                let chartData;
                let chartOptions;

                let xMin = 0, xMax = 10;
                let yMin = 0, yMax = 10;
                
                const currentModelDefaultData = modelStates[modelName].defaultData;
                const parsedDefaultData = parseDataInput(currentModelDefaultData, modelName.includes('classifier') ? 'classifier' : 'regression');
                
                if (parsedDefaultData.length > 0) {
                    const allX = parsedDefaultData.map(p => p.x);
                    const allY = parsedDefaultData.map(p => p.y);
                    
                    xMin = Math.min(...allX);
                    xMax = Math.max(...allX);
                    yMin = Math.min(...allY);
                    yMax = Math.max(...allY);

                    xMin = Math.floor(xMin) - 1;
                    xMax = Math.ceil(xMax) + 1;
                    yMin = Math.floor(yMin) - 1;
                    yMax = Math.ceil(yMax) + 1;

                    if (xMax <= xMin) xMax = xMin + 10;
                    if (yMax <= yMin) yMax = yMin + 10;

                    if (xMin > 0 && parsedDefaultData.every(p => p.x >= 0)) xMin = 0;
                    if (yMin > 0 && parsedDefaultData.every(p => p.y >= 0)) yMin = 0;
                }

                if (modelName === 'simple-linear-regression') {
                    const parsedData = parseDataInput(dataInputElem.value, 'regression');
                    modelStates[modelName].data = parsedData;
                    chartData = {
                        datasets: [
                            { label: 'Data Points', data: parsedData, backgroundColor: 'rgba(75, 192, 192, 0.6)', pointRadius: 6 },
                            { label: 'Regression Line', data: [], borderColor: 'red', type: 'line', fill: false, pointRadius: 0, borderWidth: 2 }
                        ]
                    };
                    chartOptions = { scales: { x: { type: 'linear', position: 'bottom', min: xMin, max: xMax }, y: { type: 'linear', position: 'left', min: yMin, max: yMax } } };
                } else if (modelName === 'simple-classifier' || modelName === 'mlp-classifier') {
                    const parsed = parseDataInput(dataInputElem.value, 'classifier');
                    modelStates[modelName].data.class0 = parsed.filter(p => p.label === 0);
                    modelStates[modelName].data.class1 = parsed.filter(p => p.label === 1);
                    chartData = {
                        datasets: [
                            { label: 'Class 0', data: modelStates[modelName].data.class0, backgroundColor: 'red', pointRadius: 6 },
                            { label: 'Class 1', data: modelStates[modelName].data.class1, backgroundColor: 'green', pointRadius: 6 }
                        ]
                    };
                    if (modelName === 'simple-classifier') {
                        chartData.datasets.push({ label: 'Decision Boundary', data: [], borderColor: 'purple', type: 'line', fill: false, pointRadius: 0, borderWidth: 2 });
                    } else if (modelName === 'mlp-classifier') {
                        chartData.datasets.push(
                            { label: 'Boundary Class 0', data: [], backgroundColor: 'rgba(255, 0, 0, 0.1)', pointRadius: 2, showLine: false },
                            { label: 'Boundary Class 1', data: [], backgroundColor: 'rgba(0, 128, 0, 0.1)', pointRadius: 2, showLine: false }
                        );
                    }
                    chartOptions = { scales: { x: { type: 'linear', position: 'bottom', min: xMin, max: xMax }, y: { type: 'linear', position: 'left', min: yMin, max: yMax } } };
                } else if (modelName === 'mlp-regression') {
                    const parsedData = parseDataInput(dataInputElem.value, 'regression');
                    modelStates[modelName].data = parsedData;
                    chartData = {
                        datasets: [
                            { label: 'Data Points', data: parsedData, backgroundColor: 'rgba(75, 192, 192, 0.6)', pointRadius: 6 },
                            { label: 'MLP Prediction', data: [], borderColor: 'blue', type: 'line', fill: false, pointRadius: 0, borderWidth: 2 }
                        ]
                    };
                    chartOptions = { scales: { x: { type: 'linear', position: 'bottom', min: xMin, max: xMax }, y: { type: 'linear', position: 'left', min: yMin, max: yMax } } };
                } else if (modelName === 'kmeans') {
                    const parsedData = parseDataInput(dataInputElem.value, 'regression');
                    modelStates[modelName].data = parsedData;
                    chartData = {
                        datasets: [
                            { label: 'Data Points', data: parsedData, backgroundColor: 'rgba(75, 192, 192, 0.6)', pointRadius: 6 }
                        ]
                    };
                    chartOptions = { scales: { x: { type: 'linear', position: 'bottom', min: xMin, max: xMax }, y: { type: 'linear', position: 'left', min: yMin, max: yMax } } };
                }

                renderChart(`chart-${modelName}`, 'scatter', chartData, chartOptions, modelName);

                const trainButton = document.getElementById(`button-train-${modelName}`);
                if (trainButton) trainButton.addEventListener('click', () => handleTrain(modelName));

                const resetButton = document.getElementById(`button-reset-${modelName}`);
                if (resetButton) resetButton.addEventListener('click', () => handleReset(modelName));

                dataInputElem.addEventListener('input', () => {
                    try {
                        if (modelName === 'simple-classifier' || modelName === 'mlp-classifier') {
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
                        if (modelStates[modelName].chart.data.datasets[3]) {
                            modelStates[modelName].chart.data.datasets[3].data = [];
                        }
                        modelStates[modelName].chart.update();
                    } catch (e) {
                        console.warn(`Invalid data format for ${modelName}:`, e.message);
                    }
                });
                
                document.getElementById(`display-${modelName}-loss`).textContent = 'N/A';
                const displayWeightsElem = document.getElementById(`display-${modelName}-weights`);
                if (displayWeightsElem) {
                    displayWeightsElem.textContent = 'N/A';
                }
                const displayBoundaryElem = document.getElementById(`display-${modelName}-boundary`);
                if (displayBoundaryElem) {
                    displayBoundaryElem.textContent = 'N/A';
                }

            }, 0);

        } catch (error) {
            console.error('Error fetching template:', error);
            modelContentArea.innerHTML = `<h2>Error loading template for ${modelName}.</h2>`;
            return; 
        }
    }

    if (tabButtons.length > 0) {
        activateTab(tabButtons[0].dataset.model);
    }
});