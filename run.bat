@echo off
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe BrainTumor.py
) else (
    python BrainTumor.py
)
pause