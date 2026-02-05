from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import scipy.io as sio
import os
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('experiments', exist_ok=True)

# Global variables to store training history
training_history = {}

# Global variables for preprocessing statistics (for normalization)
preprocessing_stats = {}


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def normalize_channel_coefficients(H_real, H_imag, fit=True, stats=None):
    """
    Normalize channel coefficients using z-score normalization.
    
    Args:
        H_real, H_imag: Real and imaginary parts of channel
        fit: If True, compute normalization stats; if False, use provided stats
        stats: Dictionary with 'mean' and 'std' for normalization (if fit=False)
    
    Returns:
        H_real_norm, H_imag_norm: Normalized channel coefficients
        stats: Dictionary with normalization statistics
    """
    H_combined = np.concatenate([H_real, H_imag], axis=-1)
    
    if fit:
        mean = np.mean(H_combined, axis=0, keepdims=True)
        std = np.std(H_combined, axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)  # Avoid division by zero
        stats = {'mean': mean, 'std': std}
    else:
        mean = stats['mean']
        std = stats['std']
    
    H_norm = (H_combined - mean) / std
    
    # Split back into real and imaginary
    split_idx = H_real.shape[-1]
    H_real_norm = H_norm[..., :split_idx]
    H_imag_norm = H_norm[..., split_idx:]
    
    return H_real_norm, H_imag_norm, stats


def preprocess_data(mat_data, normalize=True, fit_stats=True):
    """
    Preprocess channel data: normalize channel coefficients and convert complex to real-imaginary pairs.
    
    Args:
        mat_data: Dictionary with channel data
        normalize: Whether to normalize channel coefficients
        fit_stats: If True, compute normalization stats; if False, use global stats
    
    Returns:
        Preprocessed data dictionary with normalized values
    """
    H_real = mat_data['H_real'].copy()
    H_imag = mat_data['H_imag'].copy()
    Y_real = mat_data['Y_real'].copy()
    Y_imag = mat_data['Y_imag'].copy()
    
    # Normalize channel coefficients
    if normalize:
        if fit_stats:
            H_real_norm, H_imag_norm, stats = normalize_channel_coefficients(H_real, H_imag, fit=True)
            preprocessing_stats['channel'] = stats
        else:
            stats = preprocessing_stats.get('channel', None)
            if stats is None:
                H_real_norm, H_imag_norm, stats = normalize_channel_coefficients(H_real, H_imag, fit=True)
                preprocessing_stats['channel'] = stats
            else:
                H_real_norm, H_imag_norm, _ = normalize_channel_coefficients(H_real, H_imag, fit=False, stats=stats)
    else:
        H_real_norm, H_imag_norm = H_real, H_imag
    
    # Complex values already converted to real-imaginary pairs in the data structure
    # Return preprocessed data
    preprocessed = {
        'H_real': H_real_norm,
        'H_imag': H_imag_norm,
        'Y_real': Y_real,
        'Y_imag': Y_imag,
    }
    
    if 'Y_clean_real' in mat_data:
        preprocessed['Y_clean_real'] = mat_data['Y_clean_real'].copy()
        preprocessed['Y_clean_imag'] = mat_data['Y_clean_imag'].copy()
    
    if 'pilot_indices' in mat_data:
        preprocessed['pilot_indices'] = mat_data['pilot_indices']
    
    if 'channel_type' in mat_data:
        preprocessed['channel_type'] = mat_data['channel_type']
    
    return preprocessed


# ============================================================================
# CLASSICAL BASELINE ALGORITHMS
# ============================================================================

def least_squares_estimation(Y_pilot, X_pilot):
    """
    Least Squares (LS) channel estimation.
    
    For Y = H * X + N, LS estimate is: H_LS = Y / X (element-wise division)
    
    Args:
        Y_pilot: Received pilot signals (num_antennas, num_pilots) - complex
        X_pilot: Transmitted pilot symbols (num_pilots,) - complex
    
    Returns:
        H_est: Estimated channel (num_antennas, num_pilots) - complex
    """
    # Element-wise division: H = Y / X
    H_est = Y_pilot / (X_pilot + 1e-10)  # Small epsilon to avoid division by zero
    return H_est


def mmse_estimation(Y_pilot, X_pilot, H_true, noise_power):
    """
    Minimum Mean Square Error (MMSE) channel estimation.
    
    MMSE estimate: H_MMSE = R_HH * (R_HH + sigma^2 * I)^(-1) * H_LS
    
    Args:
        Y_pilot: Received pilot signals (num_antennas, num_pilots) - complex
        X_pilot: Transmitted pilot symbols (num_pilots,) - complex
        H_true: True channel (for computing covariance) (num_antennas, num_pilots) - complex
        noise_power: Noise power (scalar)
    
    Returns:
        H_est: MMSE estimated channel (num_antennas, num_pilots) - complex
    """
    # LS estimate first
    H_ls = least_squares_estimation(Y_pilot, X_pilot)
    
    # Compute channel covariance (simplified: use sample covariance)
    # In practice, this would be known or estimated from training data
    H_flat = H_true.flatten()
    R_HH = np.outer(H_flat, H_flat.conj()) / len(H_flat)
    
    # MMSE filter
    num_antennas, num_pilots = Y_pilot.shape
    I = np.eye(num_antennas * num_pilots)
    R_HH_inv = np.linalg.pinv(R_HH + noise_power * I)
    
    # Apply MMSE filter
    H_ls_flat = H_ls.flatten()
    H_mmse_flat = R_HH @ R_HH_inv @ H_ls_flat
    H_est = H_mmse_flat.reshape(num_antennas, num_pilots)
    
    return H_est


def ar_prediction(H_sequence, order=3):
    """
    Autoregressive (AR) model for channel prediction.
    
    Predicts next channel state using AR model: H[t] = sum(a_i * H[t-i]) + e[t]
    
    Args:
        H_sequence: Sequence of channel states (num_samples, num_antennas, num_subcarriers) - complex
        order: AR model order (default: 3)
    
    Returns:
        H_pred: Predicted next channel state (num_antennas, num_subcarriers) - complex
    """
    if len(H_sequence) < order + 1:
        # Not enough data, return last known state
        return H_sequence[-1]
    
    # Flatten for AR modeling
    H_flat_seq = H_sequence.reshape(len(H_sequence), -1)  # (time, features)
    
    # Build AR model using least squares
    # H[t] = a1*H[t-1] + a2*H[t-2] + ... + ap*H[t-p]
    X_ar = []
    y_ar = []
    
    for t in range(order, len(H_flat_seq)):
        X_ar.append(H_flat_seq[t-order:t].flatten())
        y_ar.append(H_flat_seq[t])
    
    X_ar = np.array(X_ar)
    y_ar = np.array(y_ar)
    
    # Solve: y = X * a
    # For complex data, solve real and imaginary parts separately
    X_ar_real = np.concatenate([X_ar.real, -X_ar.imag, X_ar.imag, X_ar.real], axis=1)
    y_ar_real = np.concatenate([y_ar.real, y_ar.imag], axis=1)
    
    # Least squares solution
    try:
        a = np.linalg.lstsq(X_ar_real, y_ar_real, rcond=None)[0]
        
        # Predict next state
        last_states = H_flat_seq[-order:].flatten()
        last_states_real = np.concatenate([last_states.real, -last_states.imag, last_states.imag, last_states.real])
        pred_real = last_states_real @ a
        H_pred_flat = pred_real[:, 0] + 1j * pred_real[:, 1]
        H_pred = H_pred_flat.reshape(H_sequence[0].shape)
    except:
        # Fallback: return last known state
        H_pred = H_sequence[-1]
    
    return H_pred


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_nmse(H_true, H_est):
    """
    Compute Normalized Mean Square Error (NMSE).
    
    NMSE = ||H_est - H_true||^2 / ||H_true||^2
    
    Args:
        H_true: True channel values
        H_est: Estimated channel values
    
    Returns:
        nmse: Normalized MSE (scalar)
    """
    numerator = np.mean(np.abs(H_true - H_est) ** 2)
    denominator = np.mean(np.abs(H_true) ** 2)
    nmse = numerator / (denominator + 1e-10)
    return float(nmse)


def compute_ber(H_true, H_est, threshold=0.5):
    """
    Compute Bit Error Rate (BER) for channel estimation.
    
    For channel estimation, BER is approximated as the fraction of coefficients
    that differ significantly from the true values.
    
    Args:
        H_true: True channel values
        H_est: Estimated channel values
        threshold: Relative error threshold for counting errors
    
    Returns:
        ber: Bit Error Rate (scalar, 0-1)
    """
    # Normalize by true channel magnitude
    error_magnitude = np.abs(H_true - H_est)
    true_magnitude = np.abs(H_true) + 1e-10
    relative_error = error_magnitude / true_magnitude
    
    # Count errors (coefficients with relative error > threshold)
    errors = np.sum(relative_error > threshold)
    total = relative_error.size
    
    ber = errors / total if total > 0 else 0.0
    return float(ber)


def generate_channel_data_python(channel_type='CDL', num_samples=1000, snr_db=20):
    """Python-only channel generation for 6G-style channels (no MATLAB)."""
    np.random.seed(42)

    # Channel parameters
    num_antennas = 4
    num_subcarriers = 64

    if channel_type == 'CDL':
        # Clustered Delay Line model
        num_clusters = 8
        delay_spread = 100e-9  # 100ns
        angles = np.random.uniform(0, 2 * np.pi, (num_clusters, 2))
        powers = np.random.exponential(1.0, num_clusters)
        powers = powers / powers.sum()
    elif channel_type == 'TDL':
        # Tapped Delay Line model
        num_taps = 10
        delay_spread = 200e-9  # 200ns
        powers = np.exp(-np.arange(num_taps) * 0.1)
        powers = powers / powers.sum()
    elif channel_type == 'THz':
        # THz channel model (simplified)
        num_clusters = 4
        delay_spread = 50e-9  # 50ns (shorter for THz)
        powers = np.random.exponential(0.5, num_clusters)
        powers = powers / powers.sum()
    else:
        raise ValueError(f"Unsupported channel_type: {channel_type}")

    # Storage
    H_real, H_imag = [], []
    Y_real, Y_imag = [], []
    Y_clean_real, Y_clean_imag = [], []

    pilot_indices = np.arange(0, num_subcarriers, 4)
    X_pilot = np.ones(len(pilot_indices), dtype=complex)

    noise_power = 10 ** (-snr_db / 10)

    for _ in range(num_samples):
        # Generate frequency domain channel
        H_freq = np.zeros((num_antennas, num_subcarriers), dtype=complex)

        if channel_type == 'CDL':
            for k in range(num_subcarriers):
                for cluster in range(num_clusters):
                    phase = np.random.uniform(0, 2 * np.pi)
                    H_freq[:, k] += np.sqrt(powers[cluster]) * np.exp(
                        1j * (angles[cluster, 0] + phase)
                    )
        elif channel_type == 'TDL':
            for tap in range(num_taps):
                delay = tap * delay_spread / num_taps
                for k in range(num_subcarriers):
                    phase = np.random.uniform(0, 2 * np.pi)
                    H_freq[:, k] += np.sqrt(powers[tap]) * np.exp(
                        1j * (2 * np.pi * k * delay + phase)
                    )
        else:  # THz
            for cluster in range(num_clusters):
                phase = np.random.uniform(0, 2 * np.pi)
                H_freq += np.sqrt(powers[cluster]) * np.exp(1j * phase) / num_subcarriers

        # Add Doppler effect (simplified)
        doppler_shift = np.random.uniform(-0.1, 0.1)
        H_freq *= np.exp(1j * 2 * np.pi * doppler_shift * np.arange(num_subcarriers))

        # Clean received pilots (noiseless)
        Y_clean = H_freq[:, pilot_indices] * X_pilot

        # Received signal with AWGN
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(num_antennas, len(pilot_indices))
            + 1j * np.random.randn(num_antennas, len(pilot_indices))
        )
        Y_pilot = Y_clean + noise

        H_real.append(H_freq.real)
        H_imag.append(H_freq.imag)
        Y_real.append(Y_pilot.real)
        Y_imag.append(Y_pilot.imag)
        Y_clean_real.append(Y_clean.real)
        Y_clean_imag.append(Y_clean.imag)

    return {
        'H_real': np.array(H_real),
        'H_imag': np.array(H_imag),
        'Y_real': np.array(Y_real),
        'Y_imag': np.array(Y_imag),
        'Y_clean_real': np.array(Y_clean_real),
        'Y_clean_imag': np.array(Y_clean_imag),
        'pilot_indices': pilot_indices,
        'channel_type': channel_type
    }

def create_cnn_model(input_shape, output_size):
    """Enhanced CNN for Channel Estimation with multi-scale features"""
    model = models.Sequential([
        layers.Reshape((*input_shape, 1), input_shape=input_shape),
        # Multi-scale feature extraction
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        # Deep feature extraction
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        # Residual-like connection
        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        # Feature refinement
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Flatten(),
        # Dense layers for regression
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(output_size, activation='linear')  # Output size matches target
    ])
    return model

def create_lstm_model(input_shape, output_size):
    """Enhanced LSTM for Channel Prediction with bidirectional processing"""
    model = models.Sequential([
        # Bidirectional LSTM for capturing both forward and backward dependencies
        layers.Bidirectional(layers.LSTM(128, return_sequences=True), input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        # Second LSTM layer
        layers.LSTM(128, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        # Third LSTM layer for deeper temporal understanding
        layers.LSTM(64, return_sequences=False),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        # Attention mechanism (simplified via dense layers)
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(output_size, activation='linear')  # Output size matches target
    ])
    return model

def create_dnn_model(input_shape, output_size):
    """Enhanced DNN for Channel Equalization with residual connections"""
    input_layer = layers.Input(shape=input_shape)
    x = layers.Flatten()(input_layer)
    
    # First block
    x1 = layers.Dense(1024, activation='relu')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.3)(x1)
    
    # Second block with residual connection
    x2 = layers.Dense(512, activation='relu')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.3)(x2)
    # Residual connection (projection)
    x2_res = layers.Dense(512)(x1)
    x2 = layers.Add()([x2, x2_res])
    x2 = layers.Activation('relu')(x2)
    
    # Third block
    x3 = layers.Dense(256, activation='relu')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Dropout(0.3)(x3)
    
    # Fourth block with residual connection
    x4 = layers.Dense(128, activation='relu')(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Dropout(0.2)(x4)
    x4_res = layers.Dense(128)(x3)
    x4 = layers.Add()([x4, x4_res])
    x4 = layers.Activation('relu')(x4)
    
    # Final layers
    x5 = layers.Dense(64, activation='relu')(x4)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.Dropout(0.2)(x5)
    
    # Output layer - output size matches target
    output = layers.Dense(output_size, activation='linear')(x5)
    
    model = models.Model(inputs=input_layer, outputs=output)
    return model


def load_latest_model(model_type: str):
    """Load the latest saved model file for the given model type."""
    prefix = f"{model_type.lower()}_model_"
    model_dir = 'models'
    candidates = [
        f for f in os.listdir(model_dir)
        if f.startswith(prefix) and f.endswith('.h5')
    ]
    if not candidates:
        raise FileNotFoundError(f"No saved {model_type} models found in '{model_dir}'")
    latest = sorted(candidates)[-1]
    path = os.path.join(model_dir, latest)
    model = keras.models.load_model(path)
    return model, path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate_channel', methods=['POST'])
def generate_channel():
    try:
        data = request.json
        channel_type = data.get('channel_type', 'CDL')
        num_samples = int(data.get('num_samples', 1000))
        snr_db = float(data.get('snr_db', 20))

        # Always use Python channel generation (no MATLAB fallback)
        channel_data = generate_channel_data_python(channel_type, num_samples, snr_db)
        
        # Save to .mat file
        filename = f'channel_{channel_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mat'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        sio.savemat(filepath, channel_data)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'shape': {
                'H_real': list(channel_data['H_real'].shape),
                'H_imag': list(channel_data['H_imag'].shape),
                'Y_real': list(channel_data['Y_real'].shape),
                'Y_imag': list(channel_data['Y_imag'].shape)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate_matlab_script', methods=['POST'])
def generate_matlab_script():
    try:
        data = request.json
        channel_type = data.get('channel_type', 'CDL')
        num_samples = int(data.get('num_samples', 1000))
        snr_db = float(data.get('snr_db', 20))
        
        matlab_script = f"""% 6G Channel Model Generation Script
% Channel Type: {channel_type}
% Number of Samples: {num_samples}
% SNR: {snr_db} dB

clear; clc; close all;

% Parameters
numSamples = {num_samples};
numAntennas = 4;
numSubcarriers = 64;
numSymbols = 14;
SNR_dB = {snr_db};
channelType = '{channel_type}';

% Initialize arrays
H_real = zeros(numSamples, numAntennas, numSubcarriers);
H_imag = zeros(numSamples, numAntennas, numSubcarriers);
Y_real = zeros(numSamples, numAntennas, numSubcarriers/4);
Y_imag = zeros(numSamples, numAntennas, numSubcarriers/4);

% Generate channel realizations
rng(42); % For reproducibility

for sample = 1:numSamples
    % Generate frequency domain channel
    H_freq = zeros(numAntennas, numSubcarriers);
    
    if strcmp(channelType, 'CDL')
        % Clustered Delay Line model
        numClusters = 8;
        delaySpread = 100e-9;
        angles = rand(numClusters, 2) * 2 * pi;
        powers = exprnd(1.0, numClusters, 1);
        powers = powers / sum(powers);
        
        for k = 1:numSubcarriers
            for cluster = 1:numClusters
                phase = rand * 2 * pi;
                H_freq(:, k) = H_freq(:, k) + sqrt(powers(cluster)) * ...
                    exp(1j * (angles(cluster, 1) + phase));
            end
        end
        
    elseif strcmp(channelType, 'TDL')
        % Tapped Delay Line model
        numTaps = 10;
        delaySpread = 200e-9;
        powers = exp(-(0:numTaps-1)' * 0.1);
        powers = powers / sum(powers);
        
        for tap = 1:numTaps
            delay = (tap-1) * delaySpread / numTaps;
            for k = 1:numSubcarriers
                phase = rand * 2 * pi;
                H_freq(:, k) = H_freq(:, k) + sqrt(powers(tap)) * ...
                    exp(1j * (2 * pi * k * delay + phase));
            end
        end
        
    else % THz
        % THz channel model
        numClusters = 4;
        delaySpread = 50e-9;
        powers = exprnd(0.5, numClusters, 1);
        powers = powers / sum(powers);
        
        for cluster = 1:numClusters
            phase = rand * 2 * pi;
            H_freq = H_freq + sqrt(powers(cluster)) * ...
                exp(1j * phase) / numSubcarriers;
        end
    end
    
    % Add Doppler effect
    dopplerShift = (rand - 0.5) * 0.2;
    H_freq = H_freq .* exp(1j * 2 * pi * dopplerShift * (0:numSubcarriers-1));
    
    % Generate pilot symbols
    pilotIndices = 1:4:numSubcarriers;
    X_pilot = ones(1, length(pilotIndices));
    
    % Received signal with AWGN
    noisePower = 10^(-SNR_dB/10);
    noise = sqrt(noisePower/2) * (randn(numAntennas, length(pilotIndices)) + ...
        1j * randn(numAntennas, length(pilotIndices)));
    Y_pilot = H_freq(:, pilotIndices) .* X_pilot + noise;
    
    % Store real and imaginary parts
    H_real(sample, :, :) = real(H_freq);
    H_imag(sample, :, :) = imag(H_freq);
    Y_real(sample, :, :) = real(Y_pilot);
    Y_imag(sample, :, :) = imag(Y_pilot);
end

% Save to .mat file
filename = sprintf('channel_%s_%s.mat', channelType, datestr(now, 'yyyymmdd_HHMMSS'));
save(filename, 'H_real', 'H_imag', 'Y_real', 'Y_imag', 'pilotIndices', 'channelType');

fprintf('Channel data saved to %s\\n', filename);
fprintf('Channel type: %s\\n', channelType);
fprintf('Number of samples: %d\\n', numSamples);
"""
        
        return jsonify({
            'success': True,
            'script': matlab_script
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train_model', methods=['POST'])
def train_model():
    try:
        data = request.json
        model_type = data.get('model_type', 'CNN')
        data_file = data.get('data_file')
        epochs = int(data.get('epochs', 50))
        batch_size = int(data.get('batch_size', 32))
        learning_rate = float(data.get('learning_rate', 0.001))
        
        # Load data
        if data_file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_file)
            mat_data = sio.loadmat(filepath)
        else:
            # Generate default data in Python only
            mat_data = generate_channel_data_python('CDL', 1000, 20)

        # Preprocess data (normalization, complex-to-real conversion already done)
        normalize = data.get('normalize', True)
        mat_data = preprocess_data(mat_data, normalize=normalize, fit_stats=True)

        # Prepare data depending on model type
        if 'Y_real' not in mat_data:
            return jsonify({'success': False, 'error': 'Invalid data format'}), 400

        Y_real = mat_data['Y_real']
        Y_imag = mat_data['Y_imag']

        if model_type == 'DNN':
            # DNN: equalization (noisy Y → clean Y)
            if 'Y_clean_real' not in mat_data or 'Y_clean_imag' not in mat_data:
                return jsonify({'success': False, 'error': 'Clean targets not available for DNN'}), 400
            Y_clean_real = mat_data['Y_clean_real']
            Y_clean_imag = mat_data['Y_clean_imag']

            X = np.concatenate([Y_real, Y_imag], axis=-1)  # noisy received pilots
            y = np.concatenate([Y_clean_real, Y_clean_imag], axis=-1)  # clean (equalized) pilots
        else:
            # CNN / LSTM: channel estimation / prediction (noisy Y → H)
            if 'H_real' not in mat_data or 'H_imag' not in mat_data:
                return jsonify({'success': False, 'error': 'Channel tensors not available'}), 400
            H_real = mat_data['H_real']
            H_imag = mat_data['H_imag']

            X = np.concatenate([Y_real, Y_imag], axis=-1)  # noisy received pilots
            y = np.concatenate([H_real, H_imag], axis=-1)  # true channel
        
        # Reshape for different models
        if model_type == 'CNN':
            # CNN expects (samples, features, channels)
            X = X.reshape(X.shape[0], -1, 1)
            y = y.reshape(y.shape[0], -1)
            input_shape = (X.shape[1],)
            output_size = y.shape[1]  # Target output size
            model = create_cnn_model(input_shape, output_size)
        elif model_type == 'LSTM':
            # LSTM expects (samples, timesteps, features)
            X = X.reshape(X.shape[0], X.shape[1], -1)
            y = y.reshape(y.shape[0], -1)
            input_shape = (X.shape[1], X.shape[2])
            output_size = y.shape[1]  # Target output size
            model = create_lstm_model(input_shape, output_size)
        else:  # DNN
            X = X.reshape(X.shape[0], -1)
            y = y.reshape(y.shape[0], -1)
            input_shape = (X.shape[1],)
            output_size = y.shape[1]  # Target output size
            model = create_dnn_model(input_shape, output_size)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save model
        model_filename = f'{model_type.lower()}_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
        model_path = os.path.join('models', model_filename)
        model.save(model_path)
        
        # Store history
        training_history[model_type] = {
            'history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'mae': [float(x) for x in history.history['mae']],
                'val_mae': [float(x) for x in history.history['val_mae']]
            },
            'model_path': model_path,
            'model_type': model_type
        }
        
        # Evaluate
        test_loss, test_mae, test_mse = model.evaluate(X_val, y_val, verbose=0)
        
        # Compute additional metrics (NMSE, BER)
        y_pred = model.predict(X_val, verbose=0)
        
        # For channel estimation models (CNN, LSTM), compute NMSE and BER
        if model_type in ['CNN', 'LSTM']:
            # Reshape predictions and targets back to channel shape
            y_true_reshaped = y_val.reshape(y_val.shape[0], -1)
            y_pred_reshaped = y_pred.reshape(y_pred.shape[0], -1)
            
            # Compute NMSE and BER
            test_nmse = compute_nmse(y_true_reshaped, y_pred_reshaped)
            test_ber = compute_ber(y_true_reshaped, y_pred_reshaped)
        else:
            # For DNN (equalization), compute NMSE relative to clean signal
            test_nmse = compute_nmse(y_val, y_pred)
            test_ber = compute_ber(y_val, y_pred)
        
        return jsonify({
            'success': True,
            'model_type': model_type,
            'model_path': model_path,
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'test_mse': float(test_mse),
            'test_nmse': float(test_nmse),
            'test_ber': float(test_ber),
            'epochs_trained': len(history.history['loss'])
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/run_pipeline', methods=['POST'])
def run_pipeline():
    """
    End-to-end pipeline (pure Python):
      1. Generate synthetic channel data.
      2. CNN: channel estimation (Y -> H_est).
      3. LSTM: channel prediction (Y sequence -> H_pred).
      4. DNN: equalization (noisy Y -> clean Y_eq).
    Returns shapes, MSE metrics, and sample slices for UI display.
    """
    try:
        data = request.json or {}
        channel_type = data.get('channel_type', 'CDL')
        num_samples = int(data.get('num_samples', 128))
        snr_db = float(data.get('snr_db', 20))
        samples_to_visualize = int(data.get('samples_to_visualize', 16))

        # 1) Channel generation (Python only)
        channel_data = generate_channel_data_python(channel_type, num_samples, snr_db)

        H_real = channel_data['H_real']
        H_imag = channel_data['H_imag']
        Y_real = channel_data['Y_real']
        Y_imag = channel_data['Y_imag']
        Y_clean_real = channel_data['Y_clean_real']
        Y_clean_imag = channel_data['Y_clean_imag']

        # Build tensors
        H_true = np.concatenate([H_real, H_imag], axis=-1)
        Y_obs = np.concatenate([Y_real, Y_imag], axis=-1)
        Y_clean = np.concatenate([Y_clean_real, Y_clean_imag], axis=-1)

        n = min(samples_to_visualize, H_true.shape[0])
        H_true_n = H_true[:n]
        Y_obs_n = Y_obs[:n]
        Y_clean_n = Y_clean[:n]

        # 2) CNN – channel estimation
        cnn_model, cnn_path = load_latest_model('CNN')
        X_cnn = Y_obs_n.reshape(Y_obs_n.shape[0], -1, 1)
        H_est = cnn_model.predict(X_cnn, verbose=0)

        H_true_flat = H_true_n.reshape(n, -1)
        H_est_flat = H_est.reshape(n, -1)
        cnn_mse = float(np.mean((H_est_flat - H_true_flat) ** 2))
        cnn_nmse = compute_nmse(H_true_flat, H_est_flat)
        cnn_ber = compute_ber(H_true_flat, H_est_flat)

        # 3) LSTM – channel prediction
        lstm_model, lstm_path = load_latest_model('LSTM')
        X_lstm = Y_obs_n.reshape(Y_obs_n.shape[0], Y_obs_n.shape[1], -1)
        H_pred = lstm_model.predict(X_lstm, verbose=0)
        H_pred_flat = H_pred.reshape(n, -1)
        lstm_mse = float(np.mean((H_pred_flat - H_true_flat) ** 2))
        lstm_nmse = compute_nmse(H_true_flat, H_pred_flat)
        lstm_ber = compute_ber(H_true_flat, H_pred_flat)

        # 4) DNN – equalization (noisy Y -> clean Y)
        dnn_model, dnn_path = load_latest_model('DNN')
        X_dnn = Y_obs_n.reshape(Y_obs_n.shape[0], -1)
        Y_eq = dnn_model.predict(X_dnn, verbose=0)
        Y_clean_flat = Y_clean_n.reshape(n, -1)
        Y_eq_flat = Y_eq.reshape(n, -1)
        dnn_mse = float(np.mean((Y_eq_flat - Y_clean_flat) ** 2))
        dnn_nmse = compute_nmse(Y_clean_flat, Y_eq_flat)
        dnn_ber = compute_ber(Y_clean_flat, Y_eq_flat)

        # Small slices for UI visualization
        def slice_vec(v):
            return v[: min(10, v.shape[-1])].tolist()

        sample_index = 0
        sample_outputs = {
            "true_channel_sample": slice_vec(H_true_flat[sample_index]),
            "cnn_estimated_channel_sample": slice_vec(H_est_flat[sample_index]),
            "lstm_predicted_channel_sample": slice_vec(H_pred_flat[sample_index]),
            "clean_pilot_sample": slice_vec(Y_clean_flat[sample_index]),
            "equalized_output_sample": slice_vec(Y_eq_flat[sample_index]),
        }

        return jsonify({
            "success": True,
            "params": {
                "channel_type": channel_type,
                "num_samples_generated": int(num_samples),
                "snr_db": float(snr_db),
                "samples_visualized": int(n),
            },
            "shapes": {
                "H_true": list(H_true_n.shape),
                "Y_observed": list(Y_obs_n.shape),
                "Y_clean": list(Y_clean_n.shape),
            },
            "metrics": {
                "cnn": {
                    "mse": cnn_mse,
                    "nmse": cnn_nmse,
                    "ber": cnn_ber
                },
                "lstm": {
                    "mse": lstm_mse,
                    "nmse": lstm_nmse,
                    "ber": lstm_ber
                },
                "dnn": {
                    "mse": dnn_mse,
                    "nmse": dnn_nmse,
                    "ber": dnn_ber
                }
            },
            "models_used": {
                "cnn_model_path": cnn_path,
                "lstm_model_path": lstm_path,
                "dnn_model_path": dnn_path,
            },
            "sample_outputs": sample_outputs,
        })
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        import traceback
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500

@app.route('/api/compare_performance', methods=['POST'])
def compare_performance():
    """
    Compare ML models (CNN, LSTM) with classical baselines (LS, MMSE, AR).
    
    For channel estimation: Compare CNN/LSTM vs LS/MMSE
    For channel prediction: Compare LSTM vs AR model
    
    Returns comprehensive metrics: MSE, NMSE, BER for all methods.
    """
    try:
        data = request.json or {}
        channel_type = data.get('channel_type', 'CDL')
        num_samples = int(data.get('num_samples', 200))
        snr_db = float(data.get('snr_db', 20))
        
        # Generate test data
        channel_data = generate_channel_data_python(channel_type, num_samples, snr_db)
        
        H_real = channel_data['H_real']
        H_imag = channel_data['H_imag']
        Y_real = channel_data['Y_real']
        Y_imag = channel_data['Y_imag']
        pilot_indices = channel_data['pilot_indices']
        
        # Convert to complex for classical algorithms
        H_true_complex = H_real + 1j * H_imag  # (num_samples, num_antennas, num_subcarriers)
        Y_pilot_complex = Y_real + 1j * Y_imag  # (num_samples, num_antennas, num_pilots)
        X_pilot = np.ones(len(pilot_indices), dtype=complex)
        
        noise_power = 10 ** (-snr_db / 10)
        
        # Prepare data for ML models (real-imaginary pairs)
        H_true_ml = np.concatenate([H_real, H_imag], axis=-1)
        Y_obs_ml = np.concatenate([Y_real, Y_imag], axis=-1)
        
        results = {
            'channel_estimation': {},
            'channel_prediction': {},
            'params': {
                'channel_type': channel_type,
                'num_samples': num_samples,
                'snr_db': snr_db
            }
        }
        
        # ====================================================================
        # CHANNEL ESTIMATION COMPARISON: CNN vs LS vs MMSE
        # ====================================================================
        
        # Classical LS estimation
        H_ls_list = []
        for i in range(num_samples):
            H_ls = least_squares_estimation(Y_pilot_complex[i], X_pilot)
            # Interpolate from pilot positions to all subcarriers (simplified: repeat)
            H_ls_full = np.zeros_like(H_true_complex[i])
            for idx, pilot_idx in enumerate(pilot_indices):
                H_ls_full[:, pilot_idx] = H_ls[:, idx]
            # Fill non-pilot positions with nearest neighbor
            for k in range(H_ls_full.shape[1]):
                if k not in pilot_indices:
                    nearest_pilot = pilot_indices[np.argmin(np.abs(pilot_indices - k))]
                    pilot_idx_in_list = np.where(pilot_indices == nearest_pilot)[0][0]
                    H_ls_full[:, k] = H_ls[:, pilot_idx_in_list]
            H_ls_list.append(H_ls_full)
        H_ls_all = np.array(H_ls_list)
        
        # Classical MMSE estimation
        H_mmse_list = []
        for i in range(num_samples):
            H_mmse = mmse_estimation(Y_pilot_complex[i], X_pilot, H_true_complex[i], noise_power)
            # Interpolate similar to LS
            H_mmse_full = np.zeros_like(H_true_complex[i])
            for idx, pilot_idx in enumerate(pilot_indices):
                H_mmse_full[:, pilot_idx] = H_mmse[:, idx]
            for k in range(H_mmse_full.shape[1]):
                if k not in pilot_indices:
                    nearest_pilot = pilot_indices[np.argmin(np.abs(pilot_indices - k))]
                    pilot_idx_in_list = np.where(pilot_indices == nearest_pilot)[0][0]
                    H_mmse_full[:, k] = H_mmse[:, pilot_idx_in_list]
            H_mmse_list.append(H_mmse_full)
        H_mmse_all = np.array(H_mmse_list)
        
        # Convert to real-imaginary for metric computation
        H_ls_ml = np.concatenate([H_ls_all.real, H_ls_all.imag], axis=-1)
        H_mmse_ml = np.concatenate([H_mmse_all.real, H_mmse_all.imag], axis=-1)
        
        H_true_flat = H_true_ml.reshape(num_samples, -1)
        H_ls_flat = H_ls_ml.reshape(num_samples, -1)
        H_mmse_flat = H_mmse_ml.reshape(num_samples, -1)
        
        # LS metrics
        ls_mse = float(np.mean((H_ls_flat - H_true_flat) ** 2))
        ls_nmse = compute_nmse(H_true_flat, H_ls_flat)
        ls_ber = compute_ber(H_true_flat, H_ls_flat)
        
        # MMSE metrics
        mmse_mse = float(np.mean((H_mmse_flat - H_true_flat) ** 2))
        mmse_nmse = compute_nmse(H_true_flat, H_mmse_flat)
        mmse_ber = compute_ber(H_true_flat, H_mmse_flat)
        
        results['channel_estimation']['LS'] = {
            'mse': ls_mse,
            'nmse': ls_nmse,
            'ber': ls_ber
        }
        results['channel_estimation']['MMSE'] = {
            'mse': mmse_mse,
            'nmse': mmse_nmse,
            'ber': mmse_ber
        }
        
        # CNN estimation (if model exists)
        try:
            cnn_model, _ = load_latest_model('CNN')
            X_cnn = Y_obs_ml.reshape(Y_obs_ml.shape[0], -1, 1)
            H_cnn = cnn_model.predict(X_cnn, verbose=0)
            H_cnn_flat = H_cnn.reshape(num_samples, -1)
            
            cnn_mse = float(np.mean((H_cnn_flat - H_true_flat) ** 2))
            cnn_nmse = compute_nmse(H_true_flat, H_cnn_flat)
            cnn_ber = compute_ber(H_true_flat, H_cnn_flat)
            
            results['channel_estimation']['CNN'] = {
                'mse': cnn_mse,
                'nmse': cnn_nmse,
                'ber': cnn_ber
            }
        except FileNotFoundError:
            results['channel_estimation']['CNN'] = {'error': 'CNN model not found'}
        
        # ====================================================================
        # CHANNEL PREDICTION COMPARISON: LSTM vs AR
        # ====================================================================
        
        # AR model prediction
        H_ar_list = []
        for i in range(num_samples):
            # Use previous samples as sequence
            if i < 3:
                H_ar_list.append(H_true_complex[i])
            else:
                H_seq = H_true_complex[max(0, i-10):i]  # Use last 10 samples
                H_pred_ar = ar_prediction(H_seq, order=3)
                H_ar_list.append(H_pred_ar)
        H_ar_all = np.array(H_ar_list)
        H_ar_ml = np.concatenate([H_ar_all.real, H_ar_all.imag], axis=-1)
        H_ar_flat = H_ar_ml.reshape(num_samples, -1)
        
        # AR metrics
        ar_mse = float(np.mean((H_ar_flat - H_true_flat) ** 2))
        ar_nmse = compute_nmse(H_true_flat, H_ar_flat)
        ar_ber = compute_ber(H_true_flat, H_ar_flat)
        
        results['channel_prediction']['AR'] = {
            'mse': ar_mse,
            'nmse': ar_nmse,
            'ber': ar_ber
        }
        
        # LSTM prediction (if model exists)
        try:
            lstm_model, _ = load_latest_model('LSTM')
            X_lstm = Y_obs_ml.reshape(Y_obs_ml.shape[0], Y_obs_ml.shape[1], -1)
            H_lstm = lstm_model.predict(X_lstm, verbose=0)
            H_lstm_flat = H_lstm.reshape(num_samples, -1)
            
            lstm_mse = float(np.mean((H_lstm_flat - H_true_flat) ** 2))
            lstm_nmse = compute_nmse(H_true_flat, H_lstm_flat)
            lstm_ber = compute_ber(H_true_flat, H_lstm_flat)
            
            results['channel_prediction']['LSTM'] = {
                'mse': lstm_mse,
                'nmse': lstm_nmse,
                'ber': lstm_ber
            }
        except FileNotFoundError:
            results['channel_prediction']['LSTM'] = {'error': 'LSTM model not found'}
        
        # Optionally log this comparison for research data collection
        try:
            log_record = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'type': 'comparison',
                'params': results['params'],
                'channel_estimation': results['channel_estimation'],
                'channel_prediction': results['channel_prediction'],
            }
            log_path = os.path.join('experiments', 'comparisons.jsonl')
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_record) + '\n')
        except Exception:
            # Logging must never break the API
            pass

        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/get_accuracy', methods=['GET'])
def get_accuracy():
    try:
        model_type = request.args.get('model_type', 'CNN')
        
        if model_type not in training_history:
            return jsonify({'success': False, 'error': 'Model not trained yet'}), 404
        
        history = training_history[model_type]['history']
        
        # Calculate final metrics
        final_loss = history['val_loss'][-1]
        final_mae = history['val_mae'][-1]
        
        # Calculate accuracy (1 - normalized MAE)
        max_val = max(max(history['val_mae']), 1e-6)
        accuracy = max(0, 1 - (final_mae / max_val)) * 100
        
        return jsonify({
            'success': True,
            'model_type': model_type,
            'final_loss': final_loss,
            'final_mae': final_mae,
            'accuracy': accuracy,
            'history': history
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get_training_plot', methods=['GET'])
def get_training_plot():
    try:
        model_type = request.args.get('model_type', 'CNN')
        
        if model_type not in training_history:
            return jsonify({'success': False, 'error': 'Model not trained yet'}), 404
        
        history = training_history[model_type]['history']
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(history['loss']) + 1)
        
        ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title(f'{model_type} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, history['mae'], 'b-', label='Training MAE')
        ax2.plot(epochs, history['val_mae'], 'r-', label='Validation MAE')
        ax2.set_title(f'{model_type} - Mean Absolute Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'image': img_base64
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Use port 5001 to avoid conflicts with other services on port 5000
    app.run(debug=True, host='0.0.0.0', port=5001)

