#!/bin/bash
# IDP Pipeline - Start
cd "$(dirname "$0")"
echo "🚀 Starting IDP Pipeline on http://localhost:8080"
python run.py serve --port 8080
