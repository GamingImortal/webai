Set-Location $PSScriptRoot

# Always run uvicorn with the project's virtual environment Python.
& "$PSScriptRoot\venv\Scripts\python.exe" -m uvicorn main:app --reload
