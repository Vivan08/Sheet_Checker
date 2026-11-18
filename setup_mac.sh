#!/bin/bash
set -e
cd "$(dirname "$0")"
if ! command -v brew >/dev/null 2>&1; then echo "Install Homebrew first: https://brew.sh"; exit 1; fi
brew install tesseract
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
