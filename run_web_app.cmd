@echo off
REM Navigate to the directory containing the web_app.py script
cd /d c:\Users\isaqu\Downloads\projeto-aplicado-3\scripts

REM Activate the Python virtual environment (if applicable)
REM Uncomment the next line if you are using a virtual environment
call ..\venv\Scripts\activate

REM Run the Flask application
start "" python web_app.py

REM Wait for the Flask server to start and open the browser
timeout /t 3 > nul
start http://127.0.0.1:5000
