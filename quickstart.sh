#!/bin/bash

# Quick start script for RF Anomaly Detection System
# Optimized for NVIDIA Jetson

echo "======================================"
echo "RF Anomaly Detection System"
echo "Quick Start Script"
echo "======================================"

# Check if running on Jetson
if [ -f /etc/nv_tegra_release ]; then
    echo "Detected NVIDIA Jetson platform"
    
    # Set Jetson to maximum performance
    echo "Setting Jetson to maximum performance mode..."
    sudo nvpmodel -m 0
    sudo jetson_clocks
else
    echo "Not running on Jetson - skipping performance optimization"
fi

# Check Python and CUDA
echo ""
echo "Checking environment..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Create necessary directories
mkdir -p models
mkdir -p data
mkdir -p anomalies
mkdir -p logs

# Check for trained models
echo ""
echo "Checking for trained models..."
if [ -f "models/vae_model.pth" ]; then
    echo "✓ VAE model found"
else
    echo "✗ VAE model not found - will train on first run"
fi

if [ -f "models/classifier_model.pth" ]; then
    echo "✓ Classifier model found"
else
    echo "✗ Classifier model not found"
fi

# Parse command line arguments
MODE="gui"
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-gui)
            MODE="headless"
            shift
            ;;
        --train-only)
            MODE="train"
            shift
            ;;
        --test)
            MODE="test"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo ""
echo "Starting in $MODE mode..."
echo ""

# Run appropriate mode
case $MODE in
    gui)
        echo "Starting with GUI..."
        python3 startup.py
        ;;
    headless)
        echo "Starting in headless mode..."
        python3 startup.py --no-gui
        ;;
    train)
        echo "Training mode - will collect data and train models..."
        python3 startup.py --train-only
        ;;
    test)
        echo "Test mode - running with simulated data..."
        # Start test simulator in background
        python3 test_system.py --host localhost &
        TEST_PID=$!
        sleep 2
        # Start main system
        python3 startup.py
        # Clean up
        kill $TEST_PID 2>/dev/null
        ;;
esac

echo ""
echo "System shutdown complete"