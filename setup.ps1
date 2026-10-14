# One-command setup for the ClaimCenter RAG app (Windows / PowerShell).
#
# Prerequisites you must install yourself (the ONLY two external things):
#   1. Python 3.10+        -> https://www.python.org/downloads/  (tick "Add to PATH")
#   2. Ollama + the model  -> https://ollama.com  then:  ollama pull qwen3:8b
#
# Then, from inside the RAGModel folder, run:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
# This creates a local virtual environment and installs all Python dependencies.
# It does NOT touch anything outside this folder.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "== ClaimCenter RAG setup ==" -ForegroundColor Cyan

# 1. Find a Python interpreter.
$py = $null
foreach ($c in @("python", "py", "python3")) {
    try { & $c --version *> $null; if ($LASTEXITCODE -eq 0) { $py = $c; break } } catch {}
}
if (-not $py) {
    Write-Host "Python not found on PATH. Install Python 3.10+ from python.org (tick 'Add to PATH'), then re-run." -ForegroundColor Red
    exit 1
}
Write-Host "Using Python: $(& $py --version)"

# 2. Create the virtual environment if missing.
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment (.venv)..."
    & $py -m venv .venv
} else {
    Write-Host ".venv already exists — reusing."
}

# 3. Install dependencies.
Write-Host "Installing dependencies (this can take a few minutes the first time)..."
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt

# 4. Friendly check: is Ollama reachable?
Write-Host "`nChecking for a running Ollama server..."
try {
    Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 | Out-Null
    Write-Host "Ollama is running." -ForegroundColor Green
} catch {
    Write-Host "Ollama not reachable at http://localhost:11434." -ForegroundColor Yellow
    Write-Host "Install it from https://ollama.com, then run:  ollama pull qwen3:8b" -ForegroundColor Yellow
}

Write-Host "`nSetup complete. To start the app:" -ForegroundColor Cyan
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "    streamlit run app.py"
