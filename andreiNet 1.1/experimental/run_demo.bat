@echo off
echo Compiling Cross-Entropy Demo...
g++ demoCrossEnt.cpp -o demoCrossEnt

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed! Exiting...
    pause
    exit /b %ERRORLEVEL%
)

echo Running Cross-Entropy Demo...
demoCrossEnt.exe

echo Running Visualization...
python visualize_ce.py

echo Demo Complete! Check the output/visuals directory for results.
pause
