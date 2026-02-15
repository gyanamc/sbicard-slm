#!/bin/bash

# Update and install system dependencies
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (ensure CUDA version matches EC2 instance, typically 12.1 for A10G)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers datasets peft bitsandbytes accelertate scipy safetensors trl pyyaml

# Clone repo (or assume files are uploaded)
# echo "Please upload your project files or clone the repo."

echo "Environment setup complete. Activate with 'source venv/bin/activate'"
