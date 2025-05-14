@echo off
REM Title for the command prompt window
title Running HistFlix Web Application

REM Navigate to the directory containing the script
cd /d "%~dp0" || (
    echo [ERROR] Failed to navigate to the project directory.
    pause
    exit /b 1
)

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    pause
    exit /b 1
)

REM Activate the Python virtual environment (if applicable)
if exist venv\Scripts\activate (
    call venv\Scripts\activate || (
        echo [ERROR] Failed to activate the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [WARNING] Virtual environment not found. Running without it.
)

REM Run the Flask application and capture errors
echo Starting the Flask application...

start "" python web\web_app.py > web_app.log 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] The Flask application encountered an error. Check web_app.log for more details.
    pause
    exit /b 1
)

REM Wait for the Flask server to start and open the browser
timeout /t 3 > nul
start http://127.0.0.1:5000 || (
    echo [ERROR] Failed to open the browser.
    pause
    exit /b 1
)

REM Success message
echo [INFO] The Flask application is running. You can access it at http://127.0.0.1:5000
pause