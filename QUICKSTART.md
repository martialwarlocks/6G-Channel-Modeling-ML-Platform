# Quick Start Guide

## 🚀 Fastest Way to Get Started

### Option 1: Using the Startup Script (Recommended)

```bash
cd 6g-channel-ml-web
./run.sh
```

This script will:
- Create a virtual environment (if needed)
- Install all dependencies
- Start the web server

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py
```

## 📱 Using the Web Interface

1. **Open your browser** and go to: `http://localhost:5001`

2. **Generate Channel Data** (MATLAB Channel Generation Tab):
   - Select channel type (CDL/TDL/THz)
   - Set number of samples (e.g., 1000)
   - Set SNR in dB (e.g., 20)
   - Click "Generate Channel Data" or "Generate MATLAB Script"

3. **Train a Model** (Model Training Tab):
   - Select model type (CNN/LSTM/DNN)
   - Configure training parameters
   - Click "Start Training"
   - Wait for training to complete (progress bar will show)

4. **View Results** (Accuracy & Results Tab):
   - Select the trained model
   - Click "Load Results"
   - View metrics and training plots

## 🎯 Example Workflow

1. Generate CDL channel data with 1000 samples
2. Train CNN model with 50 epochs
3. Check accuracy and view training plots
4. Repeat with LSTM and DNN models

## 💡 Tips

- Start with smaller datasets (500-1000 samples) for faster training
- Use default parameters first, then tune based on results
- The platform auto-generates data if no file is specified
- Models are automatically saved in the `models/` directory

## 🐛 Troubleshooting

**Port 5000 already in use?**
- Edit `app.py` and change the port in the last line:
  ```python
  app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
  ```

**Import errors?**
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

**Training is slow?**
- Reduce batch size or number of epochs
- Use fewer samples for initial testing

