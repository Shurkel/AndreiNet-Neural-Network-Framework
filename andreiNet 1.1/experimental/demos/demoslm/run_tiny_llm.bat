@echo off
echo Running Tiny Language Model Demo with demo_text.txt...
echo.

:: Check if the executable exists
if not exist "tiny_llm_demo.exe" (
    if exist "..\tiny_llm_demo.exe" (
        set "EXECUTABLE=..\tiny_llm_demo.exe"
    ) else (
        echo ERROR: Could not find tiny_llm_demo.exe
        echo Please make sure you're running this from the correct directory.
        pause
        exit /b 1
    )
) else (
    set "EXECUTABLE=tiny_llm_demo.exe"
)

:: Check if the text file exists
if not exist "demo_text.txt" (
    if exist "..\demo_text.txt" (
        set "TEXTFILE=..\demo_text.txt"
    ) else (
        echo ERROR: Could not find demo_text.txt
        echo Please make sure demo_text.txt exists in this directory or parent directory.
        pause
        exit /b 1
    )
) else (
    set "TEXTFILE=demo_text.txt"
)

:: Process command-line arguments
if "%1"=="--save" (
    if "%3"=="--temperature" (
        echo Training the model, saving weights to %2, and generating with temperature %4...
        %EXECUTABLE% %TEXTFILE% --save %2 --temperature %4
    ) else (
        echo Training the model and saving weights to %2...
        %EXECUTABLE% %TEXTFILE% --save %2
    )
) else if "%1"=="--load" (
    if "%3"=="--temperature" (
        echo Loading weights from %2 and generating text with temperature %4...
        %EXECUTABLE% %TEXTFILE% --load %2 --temperature %4
    ) else (
        echo Loading weights from %2 and generating text...
        %EXECUTABLE% %TEXTFILE% --load %2
    )
) else if "%1"=="--temperature" (
    echo Generating text with temperature %2...
    %EXECUTABLE% %TEXTFILE% --temperature %2
) else if "%1"=="--help" (
    echo Tiny LLM Demo - Usage Options:
    echo.
    echo run_tiny_llm.bat                                   : Run with default settings
    echo run_tiny_llm.bat --save filename.txt               : Train and save weights
    echo run_tiny_llm.bat --load filename.txt               : Load weights and skip training
    echo run_tiny_llm.bat --temperature 1.5                 : Set text randomness (0.5-1.5 recommended)
    echo run_tiny_llm.bat --load file.txt --temperature 1.5 : Load weights with custom temperature
    echo run_tiny_llm.bat --help                            : Show this help message
) else (
    %EXECUTABLE% %TEXTFILE%
)

echo.
echo Demo completed. Press any key to exit...
pause > nul
