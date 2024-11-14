let model;

// Load the model asynchronously
async function loadModel() {
    try {
        model = await tf.loadLayersModel('loan_model/model.json'); // Check if this path is correct
        console.log("Model loaded successfully.");
    } catch (error) {
        console.error("Error loading model:", error);
        document.getElementById("result").innerText = "Error loading model. Check the console for details.";
    }
}

// Run `loadModel` when the page loads
window.onload = loadModel;

// Retrieve form data and ensure all input values are collected properly
function getFormData() {
    const age = parseFloat(document.getElementById('age').value);
    const experience = parseFloat(document.getElementById('experience').value);
    const income = parseFloat(document.getElementById('income').value);
    const family = parseFloat(document.getElementById('family').value);
    const ccavg = parseFloat(document.getElementById('ccavg').value);
    const education = parseFloat(document.getElementById('education').value);
    const mortgage = parseFloat(document.getElementById('mortgage').value || 0); // Default to 0 if empty
    const securities = parseFloat(document.getElementById('securities').value);
    const cd = parseFloat(document.getElementById('cd').value);
    const online = parseFloat(document.getElementById('online').value);
    const creditcard = parseFloat(document.getElementById('creditcard').value);

    return [age, experience, income, family, ccavg, education, mortgage, securities, cd, online, creditcard];
}

// Predict loan approval
async function predictLoan() {
    // Ensure the model is loaded before proceeding
    if (!model) {
        console.error("Model not loaded yet.");
        document.getElementById("result").innerText = "Error: Model not loaded.";
        return;
    }

    try {
        // Retrieve input data
        const inputData = getFormData();
        console.log("Input data:", inputData);

        // Convert input data to tensor
        const inputTensor = tf.tensor2d([inputData], [1, 11]); // Shape should match (1, 11) for one sample with 11 features

        // Run the model prediction
        const predictionTensor = model.predict(inputTensor);
        const prediction = predictionTensor.dataSync()[0];

        // Dispose tensors to prevent memory leaks
        inputTensor.dispose();
        predictionTensor.dispose();

        // Display prediction result
        const result = prediction >= 0.5 ? "Loan Approved" : "Loan Not Approved";
        document.getElementById("result").innerText = result;
    } catch (error) {
        console.error("Error during prediction:", error);
        document.getElementById("result").innerText = "Error during prediction. Check console for details.";
    }
}
