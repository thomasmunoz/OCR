#!/bin/bash
# IDP Pipeline - Stop
echo "🛑 Stopping IDP Pipeline..."
pkill -f "python run.py serve" 2>/dev/null || pkill -f "uvicorn" 2>/dev/null
echo "✅ Stopped"
