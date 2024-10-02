#!/bin/bash

# Define the environment path
ENV_PATH="$PWD/.venv"

# Create a new Conda virtual environment
conda create -p $ENV_PATH python=3.11 -y

# Activate the virtual environment
conda activate $ENV_PATH

# Install basic data science packages
conda install numpy pandas matplotlib -y

# Install PyTorch and related packages (assuming you need CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 tensorflow -c pytorch -c nvidia -y

# Install scikit-learn
conda install scikit-learn -y

# GPUtil
conda install gputil -y

# Print the installed Python and pip versions
python --version
pip --version

echo "Environment setup is complete, and the required packages have been installed."
