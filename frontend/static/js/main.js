document.addEventListener('DOMContentLoaded', () => {
    const tabButtons = document.querySelectorAll('.tab-button');
    const modelContentArea = document.getElementById('model-content');

    const modelStates = {
        'linear-regression': { chart: null, data: [], defaultData: '1,2.5;2,4.1;3,5.8' },
        'simple-classifier': { chart: null, data: { class0: [], class1: [] }, defaultData: '1,2,0;3,4,1;5,1,0;6,3,1' },
        'mlp-regression': { chart: null, data: [], defaultData: '1,1;2,3;3,2;4,4;5,3;6,5;7,4;8,6;9,5' }
    };

    const modelTemplates = {
        'linear-regression': `
            <h2>Linear Regression Model</h2>
            <p>Model interactions and chart visualization will be enabled in the next stage.</p>
            <p>This is placeholder content for Linear Regression.</p>
        `,
        'simple-classifier': `
            <h2>Simple Classifier Model</h2>
            <p>Model interactions and chart visualization will be enabled in the next stage.</p>
            <p>This is placeholder content for Simple Classifier.</p>
        `,
        'mlp-regression': `
            <h2>MLP Regression Model</h2>
            <p>Model interactions and chart visualization will be enabled in the next stage.</p>
            <p>This is placeholder content for MLP Regression.</p>
        `
    };

    function setupSliderValueDisplay(modelName, sliderIdSuffix) {
        console.log(`Slider setup for ${modelName}-${sliderIdSuffix} (inactive in this stage).`);
    }

    function renderChart(canvasId, chartType, data, options, modelName) {
        console.log(`Chart rendering for ${canvasId} (inactive in this stage).`);
        return null;
    }

    function parseDataInput(inputString, type = 'regression') {
        console.log(`Parsing data input (inactive in this stage): ${inputString}`);
        return [];
    }

    async function handleTrain(modelName) {
        console.log(`Train button clicked for ${modelName}. Functionality coming in next stage.`);
        alert(`Training for ${modelName} is not yet implemented in this version. Check console for dummy message.`);
    }

    function handleReset(modelName) {
        console.log(`Reset button clicked for ${modelName}. Functionality coming in next stage.`);
        alert(`Reset for ${modelName} is not yet implemented in this version. Check console for dummy message.`);
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
    }

    if (tabButtons.length > 0) {
        activateTab(tabButtons[0].dataset.model);
    }
});