#!/bin/bash

# Title for the terminal window (optional, works in some terminals)
echo -ne "\033]0;Running HistFlix Web Application\007"

# Navigate to the directory containing the script
cd "$(dirname "$0")" || {
    echo "[ERROR] Failed to navigate to the project directory."
    exit 1
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python is not installed or not in PATH."
    exit 1
fi

# Activate the Python virtual environment (if applicable)
if [ -f venv/bin/activate ]; then
    source venv/bin/activate || {
        echo "[ERROR] Failed to activate the virtual environment."
        exit 1
    }
else
    echo "[WARNING] Virtual environment not found. Running without it."
fi

# Run the Flask application and capture errors
echo "Starting the Flask application..."
python3 -m web.web_app > logs/web_app.log 2>&1 &
FLASK_PID=$!

# Wait for the Flask server to start
sleep 3

# Open the default browser
open http://127.0.0.1:5000 || {
    echo "[ERROR] Failed to open the browser."
    exit 1
}

echo "[INFO] The Flask application is running. You can access it at http://127.0.0.1:5000"
wait $FLASK_PID