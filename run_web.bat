@echo off
echo ========================================================
echo   Launching NeuroScan AI - Brain Tumor Diagnostic Web App
echo ========================================================
echo.
echo Opening browser at http://127.0.0.1:5000 ...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:5000

if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe app.py
) else (
    python app.py
)
pause
