@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
REM Script to set up and run the AndreiNet GUI application on Windows

REM Get the directory where this script is located
set "PROJECT_ROOT=%~dp0"
IF "%PROJECT_ROOT:~-1%"=="\" SET "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo ============================================
echo  AndreiNet GUI Setup & Launch Script
echo ============================================
echo Project Root: %PROJECT_ROOT%
echo.

REM --- Configuration ---
set "VENV_NAME=venv"
set "PYTHON_APP_MODULE=python_gui.app"
set "BINDING_MODULE_NAME=andreinet_py"
set "REQUIREMENTS_FILE=requirements.txt"
REM --- End Configuration ---

set "VENV_PATH=%PROJECT_ROOT%\%VENV_NAME%"
set "VENV_ACTIVATE=%VENV_PATH%\Scripts\activate.bat"
set "VENV_PYTHON=%VENV_PATH%\Scripts\python.exe"
set "VENV_PIP=%VENV_PATH%\Scripts\pip.exe"
set "REQUIREMENTS_PATH=%PROJECT_ROOT%\%REQUIREMENTS_FILE%"
set "BINDING_PATTERN=%PROJECT_ROOT%\%BINDING_MODULE_NAME%*.pyd"
set "PYTHON_GUI_DIR=%PROJECT_ROOT%\python_gui"
set "INIT_PY_PATH=%PYTHON_GUI_DIR%\__init__.py"

REM --- 1. Determine Python command ---
echo [1] Locating Python...
set PYTHON_CMD=
python --version >nul 2>&1 && set "PYTHON_CMD=python" || (py --version >nul 2>&1 && set PYTHON_CMD=py)
IF NOT DEFINED PYTHON_CMD (
    echo ERROR: Python not found. Please install Python and add it to your PATH.
    pause
    exit /b 1
)
echo     Using: '%PYTHON_CMD%'
echo.

REM --- 2. Check/Create Virtual Environment ---
echo [2] Checking/Creating Virtual Environment ('%VENV_NAME%')...
IF NOT EXIST "%VENV_ACTIVATE%" (
    echo     Creating venv: %PYTHON_CMD% -m venv "%VENV_PATH%"
    "%PYTHON_CMD%" -m venv "%VENV_PATH%"
    IF ERRORLEVEL 1 (echo ERROR: Failed to create venv. & pause & exit /b 1)
    echo     Venv created successfully.
) ELSE ( echo     Venv found. )
echo.

REM --- 3. Activate Virtual Environment ---
echo [3] Activating Virtual Environment...
call "%VENV_ACTIVATE%"
IF NOT DEFINED VIRTUAL_ENV (echo ERROR: Failed to activate venv. Ensure venv exists and is not corrupted. & pause & exit /b 1)
IF NOT EXIST "%VENV_PYTHON%" (echo ERROR: Venv Python executable not found at "%VENV_PYTHON%"! & pause & exit /b 1)
echo     Venv activated. Python is now: "%VENV_PYTHON%"
echo.

REM --- 4. Install Dependencies ---
echo [4] Installing/Verifying Dependencies...
IF NOT EXIST "%REQUIREMENTS_PATH%" (
    echo     WARNING: "%REQUIREMENTS_FILE%" not found. Skipping dependency installation.
) ELSE (
    IF NOT EXIST "%VENV_PIP%" (echo ERROR: Venv Pip not found at "%VENV_PIP%"! & pause & exit /b 1)
    "%VENV_PIP%" install -r "%REQUIREMENTS_PATH%"
    IF ERRORLEVEL 1 (echo ERROR: Failed to install dependencies from "%REQUIREMENTS_FILE%". Check pip output. & pause & exit /b 1)
    echo     Dependencies are up to date.
)
echo.

REM --- 5. Check/Build C++ Bindings ---
echo [5] Checking/Building C++ Bindings ('%BINDING_MODULE_NAME%.pyd')...
IF NOT EXIST "%BINDING_PATTERN%" (
    echo     INFO: C++ binding not found. Attempting to build...
    IF EXIST "%PROJECT_ROOT%\setup.py" (
        "%VENV_PYTHON%" "%PROJECT_ROOT%\setup.py" build_ext --inplace
        IF ERRORLEVEL 1 (echo ERROR: Failed to build C++ extension. Check build output. & pause & exit /b 1)
        IF NOT EXIST "%BINDING_PATTERN%" (echo ERROR: Build seemed to complete, but binding file still not found. & pause & exit /b 1)
        echo     C++ extension built successfully.
    ) ELSE (echo ERROR: setup.py not found. Cannot build C++ extension. & pause & exit /b 1)
) ELSE (echo     C++ binding found.)
echo.

REM --- 6. Ensure __init__.py for the GUI package ---
echo [6] Ensuring Python GUI package structure...
IF NOT EXIST "%PYTHON_GUI_DIR%" (
    echo     Creating directory: "%PYTHON_GUI_DIR%"
    mkdir "%PYTHON_GUI_DIR%"
)
IF NOT EXIST "%INIT_PY_PATH%" (
    echo     Creating "%INIT_PY_PATH%" to make 'python_gui' a package.
    type NUL > "%INIT_PY_PATH%"
)
echo     Package structure verified.
echo.

REM --- 7. Launch Application ---
echo [7] Launching Application: %PYTHON_APP_MODULE%
echo     Command: "%VENV_PYTHON%" -m %PYTHON_APP_MODULE%
"%VENV_PYTHON%" -m %PYTHON_APP_MODULE%

IF ERRORLEVEL 1 GOTO PythonAppError

echo.
echo Application finished or was closed successfully.
GOTO End

:PythonAppError
echo.
echo ==================== ERROR ====================
echo The Python application exited with an error (code %ERRORLEVEL%).
echo This usually indicates a problem within the Python script itself.
echo Please check the Python console output above for any tracebacks or error messages.
echo ===============================================

:End
echo.
pause
ENDLOCAL