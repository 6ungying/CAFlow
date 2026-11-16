# Run from PowerShell after: conda activate flooddiff
# Example: .\scripts\train.ps1 --use-single-dem --batch-size 32
# Additional arguments are forwarded to train.py.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = if ($env:CAFLOW_PYTHON) { $env:CAFLOW_PYTHON } else { 'python' }
$gpuCount = if ($env:N_GPU) { $env:N_GPU } else { '1' }

& $python (Join-Path $projectRoot 'train.py') --n-gpu-per-node $gpuCount @args
exit $LASTEXITCODE
