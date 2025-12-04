#!/bin/bash

echo "Setting up ML environment..."

OS="$(uname -s)"
source venv/bin/activate 2>/dev/null || python3 -m venv venv && source venv/bin/activate

pip install --upgrade pip

if [[ "$OS" == "Darwin" ]]; then
    echo "Installing Mac dependencies..."
    pip install -r requirements_macos.txt
else
    echo "Installing Linux dependencies..."
    pip install -r requirements.txt
fi

echo "✅ Installation complete. Activate with: source venv/bin/activate"
