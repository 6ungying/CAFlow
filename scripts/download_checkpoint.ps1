param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('inpaint-center', 'inpaint-freeform1020', 'inpaint-freeform2030',
        'blur-gauss', 'blur-uni', 'sr4x-pool', 'sr4x-bicubic', 'jpeg-5', 'jpeg-10')]
    [string]$Name
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = if ($env:CAFLOW_PYTHON) { $env:CAFLOW_PYTHON } else { 'python' }
$resultsDir = if ($env:CAFLOW_RESULTS_DIR) { $env:CAFLOW_RESULTS_DIR } else {
    Join-Path $projectRoot 'results'
}
$folders = @{
    'inpaint-center' = '1ounBKyUezkZmjC7d3tzbBwfEBe0Y7bkM'
    'inpaint-freeform1020' = '1WGYzrxO-t4Q6XadgwWdUuh6Ue966LYKw'
    'inpaint-freeform2030' = '18Q1ZtUcl3z63v0VOSR5hVH3T5fGNgf3I'
    'blur-gauss' = '1vatmaOiEgO_Z9JQiFhHdcjiSNgnS9lBY'
    'blur-uni' = '1_U9IhgH7CxNp0_8o9F3D86sihflPGKDZ'
    'sr4x-pool' = '1VIA8haw3Oy0OqL-CeEM2Rekk92tL0gD0'
    'sr4x-bicubic' = '1QfpggccSV9llWMBrklZ6OPdlGR7qRd6F'
    'jpeg-5' = '17bPMUX5yBcL8YTN8nbfJT6WHXU6CNLd7'
    'jpeg-10' = '13on7v54W8ncQ4RlSl3PIZn0NalfiGaJd'
}

New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
Push-Location -LiteralPath $resultsDir
try {
    & $python -m gdown --folder $folders[$Name]
    $downloadExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $downloadExitCode
