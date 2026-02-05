# 6G Channel Modeling & ML Training Platform

A comprehensive web platform for generating 6G channel models and training AI models for channel estimation, prediction, and equalization.

## 🎯 Key Features

### 1. MATLAB Integration
- Auto-generates MATLAB script for realistic 6G channel models (CDL, TDL, THz)
- Simulates multipath propagation, Doppler effects, and AWGN
- Saves data in .mat format for seamless Python import
- Includes Python fallback for quick testing

### 2. Three AI Models
- **CNN for Estimation**: Conv1D layers extract spatial features from pilot signals
- **LSTM for Prediction**: 2-layer LSTM captures temporal channel evolution
- **DNN for Equalization**: 5-layer network inverts channel distortion

### 3. Complete Training Pipeline
- Automatic train/validation splitting
- Adam optimizer with learning rate scheduling
- Early stopping via ReduceLROnPlateau
- Batch normalization and dropout for regularization

### 4. End-to-End Workflow
MATLAB → Generate Channels → Python → Train Models → Evaluate → Save

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or navigate to the project directory:**
```bash
cd 6g-channel-ml-web
```

2. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Starting the Web Server

```bash
python app.py
```

The web interface will be available at: `http://localhost:5001`

### Using the Platform

#### 1. MATLAB Channel Generation Tab
- Select channel type (CDL, TDL, or THz)
- Set number of samples and SNR
- Click "Generate Channel Data" to create data using Python
- Click "Generate MATLAB Script" to get MATLAB code for offline generation

#### 2. Model Training Tab
- Select model type (CNN, LSTM, or DNN)
- Optionally specify a data file (or leave empty to auto-generate)
- Configure training parameters (epochs, batch size, learning rate)
- Click "Start Training" to begin training

#### 3. Accuracy & Results Tab
- Select a trained model
- View final metrics (Loss, MAE, Accuracy)
- View training history plots

## 🏗️ Architecture

### Backend (Flask)
- `app.py`: Main Flask application with API endpoints
- Channel generation (Python fallback)
- Model training and evaluation
- Accuracy metrics and visualization

### Frontend
- `templates/index.html`: Main HTML interface
- `static/css/style.css`: Modern, responsive styling
- `static/js/main.js`: Client-side JavaScript for interactions

### Models
- CNN: 3-layer Conv1D with batch normalization
- LSTM: 2-layer LSTM with dropout
- DNN: 5-layer fully connected network

## 📊 Model Details

### CNN (Channel Estimation)
- Input: Pilot signals (real + imaginary parts)
- Architecture: Conv1D → BatchNorm → Conv1D → BatchNorm → Conv1D → Dense
- Output: Estimated channel coefficients

### LSTM (Channel Prediction)
- Input: Temporal sequence of channel measurements
- Architecture: LSTM(128) → LSTM(64) → Dense layers
- Output: Predicted future channel state

### DNN (Channel Equalization)
- Input: Received signal with distortion
- Architecture: 5-layer fully connected network
- Output: Equalized signal

## 🔧 Configuration

You can modify the following in `app.py`:
- Default channel parameters
- Model architectures
- Training hyperparameters
- File paths

## 📝 Notes

- The platform uses Python fallback for channel generation by default
- To use actual MATLAB, install MATLAB Engine for Python and modify the code
- Generated data files are saved in the `data/` directory
- Trained models are saved in the `models/` directory

## 🐛 Troubleshooting

- **Import errors**: Make sure all dependencies are installed
- **Port already in use**: Change the port in `app.py` (last line)
- **Memory issues**: Reduce batch size or number of samples
- **Training slow**: Reduce epochs or use GPU-enabled TensorFlow

## 📄 License

This project is provided as-is for research and educational purposes.

## 🤝 Contributing

Feel free to enhance the models, add new features, or improve the UI!

