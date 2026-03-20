@echo off
chcp 65001 >nul 2>&1
title S^&P 500 Batch Analyzer - ValueLens

:: Use venv Python if available, fallback to PATH
set "PYTHON=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

:: Auto-install dependencies if missing
%PYTHON% -c "import yfinance" >nul 2>&1
if errorlevel 1 (
    echo   Installing dependencies...
    %PYTHON% -m pip install -r "%~dp0requirements.txt" >nul 2>&1
)

echo.
echo ============================================
echo   S^&P 500 Batch Analyzer - ValueLens
echo ============================================
echo.
echo   Usage: launcher-batch.bat [options]
echo   Options are passed through to batch_analyze.py:
echo     --limit N      Process top N tickers only
echo     --workers N    Parallel workers (default: 4)
echo     --delay N      Seconds between submissions (default: 0.5)
echo     --output F     Output CSV path (default: sp500_valuation.csv)
echo.

%PYTHON% "%~dp0batch_analyze.py" %*
if errorlevel 1 (
    echo.
    echo   [Note] Script exited with errors. Check output above.
)

echo.
echo ============================================
echo   Press any key to close...
echo ============================================
pause >nul
