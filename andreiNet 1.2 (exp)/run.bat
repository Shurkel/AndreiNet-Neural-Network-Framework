@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
REM Script to set up and run the AndreiNet GUI application on Windows

REM Get the directory where this script is located (should be the project root)
set "PROJECT_ROOT=%~dp0"
IF "%PROJECT_ROOT:~-1%"=="\" SET "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo ============================================
echo  AndreiNet GUI Setup ^& Launch Script
echo ============================================
echo Project directory: %PROJECT_ROOT%
echo.

REM --- Configuration ---
set "VENV_DIR_NAME=venv"
set "PYTHON_MODULE_TO_RUN=python_gui.app"
set "BINDING_FILE_MODULE_NAME=andreinet_py" 
set "REQUIREMENTS_FILE=requirements.txt"
REM --- End Configuration ---

set "VENV_PATH=%PROJECT_ROOT%\%VENV_DIR_NAME%"
set "VENV_ACTIVATE_SCRIPT=%VENV_PATH%\Scripts\activate.bat"
set "VENV_PYTHON_EXE=%VENV_PATH%\Scripts\python.exe"
set "VENV_PIP_EXE=%VENV_PATH%\Scripts\pip.exe"
set "REQUIREMENTS_FILE_PATH=%PROJECT_ROOT%\%REQUIREMENTS_FILE%"
set "BINDING_FILE_PATTERN=%PROJECT_ROOT%\%BINDING_FILE_MODULE_NAME%*.pyd"

REM --- Step 1: Determine Python command ---
echo [DEBUG] Determining Python command...
set PYTHON_CMD=
(python --version >nul 2>&1 && set PYTHON_CMD=python) || (py --version >nul 2>&1 && set PYTHON_CMD=py)
IF NOT DEFINED PYTHON_CMD (
    echo ERROR: Python command not found. Install Python and add to PATH.
    pause
    exit /b 1
)
echo [INFO] Using '%PYTHON_CMD%'.
echo.

REM --- Step 2: Check/Create Venv ---
IF NOT EXIST "%VENV_ACTIVATE_SCRIPT%" (
    echo Creating venv: %PYTHON_CMD% -m venv "%VENV_PATH%"
    "%PYTHON_CMD%" -m venv "%VENV_PATH%"
    IF ERRORLEVEL 1 (echo ERROR: Failed to create venv. & pause & exit /b 1)
    echo Venv created.
) ELSE ( echo Venv found. )
echo.

REM --- Step 3: Activate Venv ---
echo Activating venv...
call "%VENV_ACTIVATE_SCRIPT%"
IF NOT DEFINED VIRTUAL_ENV (echo ERROR: Failed to activate venv. & pause & exit /b 1)
echo Venv activated. Python: "%VENV_PYTHON_EXE%"
IF NOT EXIST "%VENV_PYTHON_EXE%" (echo ERROR: Venv Python not found! & pause & exit /b 1)
echo.

REM --- Step 4: Install Dependencies ---
IF NOT EXIST "%REQUIREMENTS_FILE_PATH%" (
    echo WARNING: "%REQUIREMENTS_FILE%" not found. Skipping deps.
) ELSE (
    echo Installing dependencies...
    IF NOT EXIST "%VENV_PIP_EXE%" (echo ERROR: Venv Pip not found! & pause & exit /b 1)
    "%VENV_PIP_EXE%" install -r "%REQUIREMENTS_FILE_PATH%"
    IF ERRORLEVEL 1 (echo ERROR: Failed to install deps. & pause & exit /b 1)
    echo Dependencies up to date.
)
echo.

REM --- Step 5: Check/Build C++ Bindings ---
IF NOT EXIST "%BINDING_FILE_PATTERN%" (
    echo INFO: C++ binding not found. Attempting to build...
    IF EXIST "%PROJECT_ROOT%\setup.py" (
        "%VENV_PYTHON_EXE%" "%PROJECT_ROOT%\setup.py" build_ext --inplace
        IF ERRORLEVEL 1 (echo ERROR: Failed to build C++ extension. & pause & exit /b 1)
        IF NOT EXIST "%BINDING_FILE_PATTERN%" (echo ERROR: Build complete but binding still not found. & pause & exit /b 1)
        echo C++ extension built.
    ) ELSE (echo WARNING: setup.py not found. Cannot build. & pause & exit /b 1)
) ELSE (echo C++ binding found.)
echo.

REM --- Step 6: Ensure __init__.py ---
IF NOT EXIST "%PROJECT_ROOT%\python_gui\" (mkdir "%PROJECT_ROOT%\python_gui\")
IF NOT EXIST "%PROJECT_ROOT%\python_gui\__init__.py" (
    echo Creating __init__.py...
    type NUL > "%PROJECT_ROOT%\python_gui\__init__.py"
)

REM --- Step 7: Launch Application (Simplified End) ---
echo Launching application module: %PYTHON_MODULE_TO_RUN%
echo [DEBUG] Running: "%VENV_PYTHON_EXE%" -m %PYTHON_MODULE_TO_RUN%
"%VENV_PYTHON_EXE%" -m %PYTHON_MODULE_TO_RUN%

REM --- Critical Change: Check ERRORLEVEL simply and exit or continue ---
IF ERRORLEVEL 1 GOTO PythonError

echo.
echo Application finished or was closed successfully (Python exit code 0).
GOTO EndScript

:PythonError
echo.
echo ERROR: Python application exited with a non-zero status (code %ERRORLEVEL%).
echo This usually indicates an error within the Python script.
echo Check the Python console output above for tracebacks or error messages from app.py.

:EndScript
pause
ENDLOCAL