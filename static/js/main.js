// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

// Generate Channel Data
async function generateChannel() {
    const channelType = document.getElementById('channel-type').value;
    const numSamples = document.getElementById('num-samples').value;
    const snrDb = document.getElementById('snr-db').value;
    const statusDiv = document.getElementById('matlab-status');
    
    statusDiv.className = 'status-message info';
    statusDiv.textContent = 'Generating channel data...';
    statusDiv.style.display = 'block';
    
    try {
        const response = await fetch('/api/generate_channel', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                channel_type: channelType,
                num_samples: numSamples,
                snr_db: snrDb
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            statusDiv.className = 'status-message success';
            statusDiv.innerHTML = `
                <strong>Success.</strong><br>
                Channel data generated: ${data.filename}<br>
                Shape: H_real ${data.shape.H_real.join('x')}, 
                Y_real ${data.shape.Y_real.join('x')}
            `;
        } else {
            throw new Error(data.error || 'Failed to generate channel data');
        }
    } catch (error) {
        statusDiv.className = 'status-message error';
        statusDiv.textContent = `Error: ${error.message}`;
    }
}

// Generate MATLAB Script
async function generateMatlabScript() {
    const channelType = document.getElementById('channel-type').value;
    const numSamples = document.getElementById('num-samples').value;
    const snrDb = document.getElementById('snr-db').value;
    const statusDiv = document.getElementById('matlab-status');
    
    statusDiv.className = 'status-message info';
    statusDiv.textContent = 'Generating MATLAB script...';
    statusDiv.style.display = 'block';
    
    try {
        const response = await fetch('/api/generate_matlab_script', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                channel_type: channelType,
                num_samples: numSamples,
                snr_db: snrDb
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            statusDiv.className = 'status-message success';
            statusDiv.textContent = 'MATLAB script generated successfully.';
            
            document.getElementById('matlab-script-content').textContent = data.script;
            document.getElementById('matlab-script-output').style.display = 'block';
        } else {
            throw new Error(data.error || 'Failed to generate MATLAB script');
        }
    } catch (error) {
        statusDiv.className = 'status-message error';
        statusDiv.textContent = `Error: ${error.message}`;
    }
}

// Copy MATLAB Script
function copyMatlabScript() {
    const scriptContent = document.getElementById('matlab-script-content').textContent;
    navigator.clipboard.writeText(scriptContent).then(() => {
        alert('MATLAB script copied to clipboard!');
    });
}

// Train Model
async function trainModel() {
    const modelType = document.getElementById('model-type').value;
    const dataFile = document.getElementById('data-file').value;
    const epochs = document.getElementById('epochs').value;
    const batchSize = document.getElementById('batch-size').value;
    const learningRate = document.getElementById('learning-rate').value;
    const statusDiv = document.getElementById('training-status');
    const progressDiv = document.getElementById('training-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    statusDiv.className = 'status-message info';
    statusDiv.textContent = `Training ${modelType} model...`;
    statusDiv.style.display = 'block';
    progressDiv.style.display = 'block';
    
    // Simulate progress (actual progress would come from server via WebSocket or polling)
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 10;
        if (progress > 90) progress = 90;
        progressFill.style.width = progress + '%';
        progressText.textContent = `Training in progress... ${Math.round(progress)}%`;
    }, 1000);
    
    try {
        const response = await fetch('/api/train_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_type: modelType,
                data_file: dataFile || null,
                epochs: epochs,
                batch_size: batchSize,
                learning_rate: learningRate
            })
        });
        
        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressText.textContent = 'Training completed!';
        
        const data = await response.json();
        
        if (data.success) {
            statusDiv.className = 'status-message success';
            let metricsHtml = `
                <strong>Training complete.</strong><br>
                Model Type: ${data.model_type}<br>
                Test Loss: ${data.test_loss.toFixed(6)}<br>
                Test MAE: ${data.test_mae.toFixed(6)}<br>
                Test MSE: ${data.test_mse.toFixed(6)}<br>
            `;
            if (data.test_nmse !== undefined) {
                metricsHtml += `Test NMSE: ${data.test_nmse.toExponential(4)}<br>`;
            }
            if (data.test_ber !== undefined) {
                metricsHtml += `Test BER: ${data.test_ber.toFixed(6)}<br>`;
            }
            metricsHtml += `
                Epochs Trained: ${data.epochs_trained}<br>
                Model saved to: ${data.model_path}
            `;
            statusDiv.innerHTML = metricsHtml;
            
            setTimeout(() => {
                progressDiv.style.display = 'none';
            }, 2000);
        } else {
            throw new Error(data.error || 'Training failed');
        }
    } catch (error) {
        clearInterval(progressInterval);
        statusDiv.className = 'status-message error';
        statusDiv.textContent = `Error: ${error.message}`;
        progressDiv.style.display = 'none';
    }
}

// Load Accuracy
async function loadAccuracy() {
    const modelType = document.getElementById('accuracy-model-type').value;
    const statusDiv = document.getElementById('accuracy-status');
    const resultsDiv = document.getElementById('accuracy-results');
    const compStatusDiv = document.getElementById('accuracy-comp-status');
    const compResultsDiv = document.getElementById('accuracy-comp-results');
    
    statusDiv.className = 'status-message info';
    statusDiv.textContent = 'Loading accuracy results...';
    statusDiv.style.display = 'block';
    if (compStatusDiv) compStatusDiv.style.display = 'none';
    if (compResultsDiv) compResultsDiv.style.display = 'none';
    
    try {
        const response = await fetch(`/api/get_accuracy?model_type=${modelType}`);
        const data = await response.json();
        
        if (data.success) {
            statusDiv.className = 'status-message success';
            statusDiv.textContent = 'Results loaded successfully!';
            
            // Update metrics
            document.getElementById('final-loss').textContent = data.final_loss.toFixed(6);
            document.getElementById('final-mae').textContent = data.final_mae.toFixed(6);
            document.getElementById('accuracy-value').textContent = data.accuracy.toFixed(2) + '%';
            
            // Load training plot
            const plotResponse = await fetch(`/api/get_training_plot?model_type=${modelType}`);
            const plotData = await plotResponse.json();
            
            if (plotData.success) {
                const plotImg = document.getElementById('training-plot');
                plotImg.src = 'data:image/png;base64,' + plotData.image;
                plotImg.style.display = 'block';
            }
            
            resultsDiv.style.display = 'block';
        } else {
            throw new Error(data.error || 'Failed to load accuracy');
        }
    } catch (error) {
        statusDiv.className = 'status-message error';
        statusDiv.textContent = `Error: ${error.message}`;
        resultsDiv.style.display = 'none';
    }
}

// Baseline comparison for selected model (within Accuracy tab)
async function compareAccuracy() {
    const modelType = document.getElementById('accuracy-model-type').value;
    const channelType = document.getElementById('accuracy-comp-channel-type').value;
    const numSamples = parseInt(document.getElementById('accuracy-comp-num-samples').value, 10);
    const snrDb = parseFloat(document.getElementById('accuracy-comp-snr-db').value);

    const statusDiv = document.getElementById('accuracy-comp-status');
    const resultsDiv = document.getElementById('accuracy-comp-results');

    statusDiv.className = 'status-message info';
    statusDiv.textContent = 'Running baseline comparison on fresh test data...';
    statusDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    resultsDiv.innerHTML = '';

    try {
        const response = await fetch('/api/compare_performance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                channel_type: channelType,
                num_samples: numSamples,
                snr_db: snrDb
            })
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Baseline comparison failed');
        }

        const results = data.results;
        const params = results.params;

        statusDiv.className = 'status-message success';
        statusDiv.textContent = 'Baseline comparison completed. See table below.';

        let sectionTitle = '';
        let methods = {};

        if (modelType === 'CNN') {
            sectionTitle = 'Channel estimation (ML CNN vs LS/MMSE)';
            methods = results.channel_estimation || {};
        } else if (modelType === 'LSTM') {
            sectionTitle = 'Channel prediction (ML LSTM vs AR)';
            methods = results.channel_prediction || {};
        } else {
            sectionTitle = 'No classical baseline defined for DNN equalisation; results shown are ML-only.';
            methods = {};
        }

        let html = `
            <div class="card" style="margin-bottom: 10px;">
                <h3>Test configuration</h3>
                <p>Channel type: <strong>${params.channel_type}</strong></p>
                <p>Number of samples: <strong>${params.num_samples}</strong></p>
                <p>SNR: <strong>${params.snr_db} dB</strong></p>
            </div>
        `;

        if (Object.keys(methods).length > 0) {
            html += `
                <div class="card">
                    <h3>${sectionTitle}</h3>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                        <thead>
                            <tr style="background: #f3f4f6; border-bottom: 2px solid #e5e7eb;">
                                <th style="padding: 8px; text-align: left;">Method</th>
                                <th style="padding: 8px; text-align: right;">MSE</th>
                                <th style="padding: 8px; text-align: right;">NMSE</th>
                                <th style="padding: 8px; text-align: right;">BER</th>
                            </tr>
                        </thead>
                        <tbody>
            `;

            for (const [method, metrics] of Object.entries(methods)) {
                const label = method === modelType ? `${method} (ML)` : `${method} (classical)`;
                if (metrics.error) {
                    html += `
                        <tr>
                            <td style="padding: 8px;"><strong>${label}</strong></td>
                            <td colspan="3" style="padding: 8px; color: #9ca3af;">${metrics.error}</td>
                        </tr>
                    `;
                } else {
                    html += `
                        <tr style="border-bottom: 1px solid #e5e7eb;">
                            <td style="padding: 8px;"><strong>${label}</strong></td>
                            <td style="padding: 8px; text-align: right;">${metrics.mse.toExponential(4)}</td>
                            <td style="padding: 8px; text-align: right;">${metrics.nmse.toExponential(4)}</td>
                            <td style="padding: 8px; text-align: right;">${metrics.ber.toFixed(6)}</td>
                        </tr>
                    `;
                }
            }

            html += `
                        </tbody>
                    </table>
                </div>
            `;
        } else {
            html += `
                <div class="card">
                    <p class="field-hint">${sectionTitle}</p>
                </div>
            `;
        }

        resultsDiv.innerHTML = html;
        resultsDiv.style.display = 'block';
    } catch (error) {
        statusDiv.className = 'status-message error';
        statusDiv.textContent = `Error: ${error.message}`;
        resultsDiv.style.display = 'none';
    }
}

// End-to-end pipeline
async function runPipeline() {
    const channelType = document.getElementById('pipeline-channel-type').value;
    const numSamples = parseInt(document.getElementById('pipeline-num-samples').value, 10);
    const snrDb = parseFloat(document.getElementById('pipeline-snr-db').value);
    const samplesToVisualize = parseInt(document.getElementById('pipeline-samples-visualize').value, 10);

    const statusDiv = document.getElementById('pipeline-status');
    const stepsDiv = document.getElementById('pipeline-steps');

    statusDiv.className = 'status-message info';
    statusDiv.textContent = 'Running end-to-end pipeline (channel generation → CNN → LSTM → DNN)...';
    statusDiv.style.display = 'block';
    stepsDiv.style.display = 'none';
    stepsDiv.innerHTML = '';

    try {
        const response = await fetch('/api/run_pipeline', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                channel_type: channelType,
                num_samples: numSamples,
                snr_db: snrDb,
                samples_to_visualize: samplesToVisualize
            })
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Pipeline failed');
        }

        const p = data.params;
        const m = data.metrics;
        const s = data.shapes;
        const o = data.sample_outputs;

        statusDiv.className = 'status-message success';
        statusDiv.textContent = 'Pipeline completed successfully. See detailed steps below.';

        stepsDiv.innerHTML = `
            <div class="card" style="margin-bottom: 12px;">
                <h3>Step 1 – Channel generation (Python)</h3>
                <p>Channel type: <strong>${p.channel_type}</strong></p>
                <p>Samples generated: <strong>${p.num_samples_generated}</strong>, SNR: <strong>${p.snr_db} dB</strong></p>
                <p>Tensor shapes: H_true ${JSON.stringify(s.H_true)}, Y_observed ${JSON.stringify(s.Y_observed)}</p>
            </div>

            <div class="card" style="margin-bottom: 12px;">
                <h3>Step 2 – CNN channel estimation</h3>
                <p>Task: Estimate the complex channel from noisy pilot observations.</p>
                <p><strong>Metrics:</strong> MSE: ${m.cnn.mse.toExponential(4)}, NMSE: ${m.cnn.nmse.toExponential(4)}, BER: ${m.cnn.ber.toFixed(6)}</p>
                <p>Sample (first few flattened coefficients):</p>
                <pre>${JSON.stringify(o.true_channel_sample)}  ← true H
${JSON.stringify(o.cnn_estimated_channel_sample)}  ← CNN estimate</pre>
            </div>

            <div class="card" style="margin-bottom: 12px;">
                <h3>Step 3 – LSTM channel prediction</h3>
                <p>Task: Predict channel behaviour using sequence modelling over the observations.</p>
                <p><strong>Metrics:</strong> MSE: ${m.lstm.mse.toExponential(4)}, NMSE: ${m.lstm.nmse.toExponential(4)}, BER: ${m.lstm.ber.toFixed(6)}</p>
                <p>Sample (first few flattened coefficients):</p>
                <pre>${JSON.stringify(o.true_channel_sample)}  ← true H
${JSON.stringify(o.lstm_predicted_channel_sample)}  ← LSTM prediction</pre>
            </div>

            <div class="card">
                <h3>Step 4 – DNN equalisation</h3>
                <p>Task: Map noisy received pilots to their clean, equalised versions.</p>
                <p><strong>Metrics:</strong> MSE: ${m.dnn.mse.toExponential(4)}, NMSE: ${m.dnn.nmse.toExponential(4)}, BER: ${m.dnn.ber.toFixed(6)}</p>
                <p>Sample (first few flattened coefficients):</p>
                <pre>${JSON.stringify(o.clean_pilot_sample)}  ← clean target
${JSON.stringify(o.equalized_output_sample)}  ← DNN equalised output</pre>
            </div>
        `;

        stepsDiv.style.display = 'block';
    } catch (error) {
        statusDiv.className = 'status-message error';
        statusDiv.textContent = `Error: ${error.message}`;
        stepsDiv.style.display = 'none';
    }
}

// Performance Comparison
async function runComparison() {
    const channelType = document.getElementById('comparison-channel-type').value;
    const numSamples = parseInt(document.getElementById('comparison-num-samples').value, 10);
    const snrDb = parseFloat(document.getElementById('comparison-snr-db').value);

    const statusDiv = document.getElementById('comparison-status');
    const resultsDiv = document.getElementById('comparison-results');

    statusDiv.className = 'status-message info';
    statusDiv.textContent = 'Running performance comparison (ML models vs classical baselines)...';
    statusDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    resultsDiv.innerHTML = '';

    try {
        const response = await fetch('/api/compare_performance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                channel_type: channelType,
                num_samples: numSamples,
                snr_db: snrDb
            })
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Comparison failed');
        }

        const results = data.results;
        const params = data.results.params;

        statusDiv.className = 'status-message success';
        statusDiv.textContent = 'Comparison completed successfully. See results below.';

        let html = `
            <div class="card" style="margin-bottom: 12px;">
                <h3>Test Configuration</h3>
                <p>Channel type: <strong>${params.channel_type}</strong></p>
                <p>Number of samples: <strong>${params.num_samples}</strong></p>
                <p>SNR: <strong>${params.snr_db} dB</strong></p>
            </div>
        `;

        // Channel Estimation Comparison
        if (results.channel_estimation) {
            html += `
                <div class="card" style="margin-bottom: 12px;">
                    <h3>Channel Estimation Comparison</h3>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                        <thead>
                            <tr style="background: #f3f4f6; border-bottom: 2px solid #e5e7eb;">
                                <th style="padding: 8px; text-align: left;">Method</th>
                                <th style="padding: 8px; text-align: right;">MSE</th>
                                <th style="padding: 8px; text-align: right;">NMSE</th>
                                <th style="padding: 8px; text-align: right;">BER</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            for (const [method, metrics] of Object.entries(results.channel_estimation)) {
                if (metrics.error) {
                    html += `
                        <tr>
                            <td style="padding: 8px;"><strong>${method}</strong></td>
                            <td colspan="3" style="padding: 8px; color: #9ca3af;">${metrics.error}</td>
                        </tr>
                    `;
                } else {
                    html += `
                        <tr style="border-bottom: 1px solid #e5e7eb;">
                            <td style="padding: 8px;"><strong>${method}</strong></td>
                            <td style="padding: 8px; text-align: right;">${metrics.mse.toExponential(4)}</td>
                            <td style="padding: 8px; text-align: right;">${metrics.nmse.toExponential(4)}</td>
                            <td style="padding: 8px; text-align: right;">${metrics.ber.toFixed(6)}</td>
                        </tr>
                    `;
                }
            }
            
            html += `
                        </tbody>
                    </table>
                </div>
            `;
        }

        // Channel Prediction Comparison
        if (results.channel_prediction) {
            html += `
                <div class="card">
                    <h3>Channel Prediction Comparison</h3>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                        <thead>
                            <tr style="background: #f3f4f6; border-bottom: 2px solid #e5e7eb;">
                                <th style="padding: 8px; text-align: left;">Method</th>
                                <th style="padding: 8px; text-align: right;">MSE</th>
                                <th style="padding: 8px; text-align: right;">NMSE</th>
                                <th style="padding: 8px; text-align: right;">BER</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            for (const [method, metrics] of Object.entries(results.channel_prediction)) {
                if (metrics.error) {
                    html += `
                        <tr>
                            <td style="padding: 8px;"><strong>${method}</strong></td>
                            <td colspan="3" style="padding: 8px; color: #9ca3af;">${metrics.error}</td>
                        </tr>
                    `;
                } else {
                    html += `
                        <tr style="border-bottom: 1px solid #e5e7eb;">
                            <td style="padding: 8px;"><strong>${method}</strong></td>
                            <td style="padding: 8px; text-align: right;">${metrics.mse.toExponential(4)}</td>
                            <td style="padding: 8px; text-align: right;">${metrics.nmse.toExponential(4)}</td>
                            <td style="padding: 8px; text-align: right;">${metrics.ber.toFixed(6)}</td>
                        </tr>
                    `;
                }
            }
            
            html += `
                        </tbody>
                    </table>
                </div>
            `;
        }

        resultsDiv.innerHTML = html;
        resultsDiv.style.display = 'block';
    } catch (error) {
        statusDiv.className = 'status-message error';
        statusDiv.textContent = `Error: ${error.message}`;
        resultsDiv.style.display = 'none';
    }
}
