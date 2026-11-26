param([string]$Python = "")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not $Python) {
    $LocalPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $LocalPython) { $Python = $LocalPython } else { $Python = "python" }
}
& $Python -u (Join-Path $PSScriptRoot "run.py") all --figures
exit $LASTEXITCODE
