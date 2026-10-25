# Run in a dedicated visible PowerShell window. Re-running resumes existing data.
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$taskPython = 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe'
$taskDataRoot = 'H:\pangu_2025_data\model_input\gdas'
$taskLogDir = Join-Path $taskDataRoot '_runs'
$taskRunStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$taskLogPath = Join-Path $taskLogDir "annual_console_$taskRunStamp.log"

Set-Location -LiteralPath $taskProjectRoot
New-Item -ItemType Directory -Path $taskLogDir -Force | Out-Null
$Host.UI.RawUI.WindowTitle = 'GDAS 2025 - Download and Prepare - 370 days'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:PANGU_GDAS_ROOT = $taskDataRoot
$env:PANGU_GDAS_RAW_ROOT = Join-Path $taskDataRoot 'raw'
Start-Transcript -Path $taskLogPath | Out-Null
try {
    Write-Host 'GDAS/FNL download + conversion: 2024-12-27 to 2025-12-31, 00 UTC daily.' -ForegroundColor Cyan
    Write-Host '370 cycles. Existing valid files are skipped; interrupted downloads resume.' -ForegroundColor Cyan
    Write-Host "Raw files: $taskDataRoot\raw\YEAR"
    Write-Host "Prepared:  $taskDataRoot\YEAR\YYYY-MM-DD-HH-MM"
    Write-Host "Log:       $taskLogPath"
    Write-Host 'Press Ctrl+C to interrupt. Run this script again to resume.'
    Write-Host ''
    & $taskPython -u -m experiment_2025.gdas run --start 2024-12-27 --end 2025-12-31 --interval-hours 24 --continue-on-error
    $taskExitCode = $LASTEXITCODE
    if ($taskExitCode -eq 0) {
        Write-Host 'All 370 cycles completed successfully.' -ForegroundColor Green
    } else {
        Write-Host "Batch ended with exit code $taskExitCode. Inspect _runs reports for failed cycles; re-run to retry." -ForegroundColor Yellow
    }
} finally {
    Stop-Transcript | Out-Null
}
