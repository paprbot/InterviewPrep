# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..."
pip install -r requirements.txt

Write-Host "Virtual environment setup complete." 