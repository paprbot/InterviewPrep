@echo off
REM Create virtual environment
if not exist venv\Scripts\python.exe (
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

echo Virtual environment setup complete. 

rmdir /s /q venv
setup_venv.bat 