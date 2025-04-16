#!/bin/bash

# Update package list
apt-get -y update

# Install Conda if it's not installed (Optional step)
# You can skip this if Conda is already installed
apt-get install -y conda

# Create environment from the environment.yml file
conda env create -f environment.yml

# Activate the Conda environment
conda activate sentiment

# Install project dependencies (if there’s a requirements.txt for any pip packages)
pip install -r requirements.txt
