# Visible progress window; Ctrl+C interrupts, rerunning resumes validated results.
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$taskPython = 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe'
$taskOutputRoot = 'H:\pangu_2025_data\model_output\experiment_2025'
$taskLogDir = Join-Path $taskOutputRoot '_runs'
$taskRunStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$taskLogPath = Join-Path $taskLogDir "era5_week_console_$taskRunStamp.log"
$taskLeads = (0..40 | ForEach-Object { $_ * 3 }) -join ','

Set-Location -LiteralPath $taskProjectRoot
New-Item -ItemType Directory -Path $taskLogDir -Force | Out-Null
$Host.UI.RawUI.WindowTitle = 'ERA5 Direct: July 1-7 2025 | 0-120h every 3h | 287 outputs'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
Start-Transcript -Path $taskLogPath | Out-Null
$taskTimer = [System.Diagnostics.Stopwatch]::StartNew()
try {
    Write-Host 'ERA5 DIRECT FORECAST TEST (inference, no model training)' -ForegroundColor Cyan
    Write-Host '2025-07-01 to 2025-07-07, 00 UTC daily; leads 0,3,...,120h.'
    Write-Host '7 initialization dates x 41 leads = 287 wind-only files.'
    Write-Host 'Input: daily ERA5 NetCDF; preserve original values.'
    Write-Host "Output: $taskOutputRoot\era5_direct\2025"
    Write-Host "Console log: $taskLogPath"
    Write-Host 'Timing: total, daily tasks, input, model loading, inference, checkpoint writes and output writes.'
    Write-Host 'A running-time message appears every 30 seconds. Ctrl+C interrupts; rerun this script to resume.'
    Write-Host ''
    & $taskPython -u -m experiment_2025 --start-date 2025-07-01 --end-date 2025-07-07 --experiments era5_direct --lead-hours $taskLeads --output-root $taskOutputRoot
    $taskExitCode = $LASTEXITCODE
    if ($taskExitCode -eq 0) {
        Write-Host 'All 7 dates completed successfully.' -ForegroundColor Green
    } else {
        Write-Host "Batch exited with code $taskExitCode. Inspect the JSON report; failed tasks retain checkpoints." -ForegroundColor Yellow
    }
} finally {
    $taskTimer.Stop()
    Write-Host "Window run elapsed: $($taskTimer.Elapsed.ToString())"
    Stop-Transcript | Out-Null
}

