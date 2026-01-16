#!/bin/bash
# Setup script for crystallized-safety

set -e

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Activating venv..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip -q

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete! Activate with:"
echo "  source .venv/bin/activate"
