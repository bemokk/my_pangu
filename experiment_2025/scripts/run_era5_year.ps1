# Annual ERA5 forecasts, concise console, full UTF-8 log. Re-run to resume.
param(
    [ValidateSet('era5_direct', 'era5_lagged_120h')]
    [string]$Experiment = 'era5_direct'
)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$taskPython = 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe'
$taskOutputRoot = 'H:\pangu_2025_data\model_output\experiment_2025'
$taskLogDir = Join-Path $taskOutputRoot '_runs'
$taskRunStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$taskLogPrefix = if ($Experiment -eq 'era5_direct') { 'era5_year_detail' } else { 'era5_lagged_year_detail' }
$taskLogPath = Join-Path $taskLogDir "${taskLogPrefix}_$taskRunStamp.log"
$taskLeads = (0..40 | ForEach-Object { $_ * 3 }) -join ','

Set-Location -LiteralPath $taskProjectRoot
New-Item -ItemType Directory -Path $taskLogDir -Force | Out-Null
$Host.UI.RawUI.WindowTitle = "$Experiment 2025 | 365 days | 0-120h / 3h"
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $OutputEncoding
Write-Host "$Experiment | 2025-01-01 to 2025-12-31 | 00 UTC" -ForegroundColor Cyan
Write-Host '365 dates x 41 leads (0-120h / 3h). Valid existing results are reused.'
if ($Experiment -eq 'era5_lagged_120h') {
    Write-Host 'ERA5 initial time: T0-120h; actual model integration: 120-240h.'
}
Write-Host "Details: $taskLogPath"
Write-Host 'Ctrl+C: interrupt; run this script again to resume.'
Write-Host ''

# One live line per initialization date, followed by one completion line.
# Unicode escapes keep this script readable by Windows PowerShell 5.1
# regardless of its default text-file encoding.
$taskStartPattern = '^\[\u4efb\u52a1\u5f00\u59cb\]\s+(?<index>\d+/\d+)\s+\S+\s+(?<date>\S+)'
$taskEndPattern = '^\[(?:\u672c\u65e5\u5b8c\u6210|\u8df3\u8fc7)\]'
$taskRunningPattern = '^\[\u8fd0\u884c\u4e2d\]'
$taskErrorPattern = '^\[\u5931\u8d25\]'
$taskRunningTag = [regex]::Unescape('[\u8fd0\u884c\u4e2d]')
$taskCompleteTag = [regex]::Unescape('[\u5df2\u5b8c\u6210]')
$taskElapsedTag = [regex]::Unescape('\u7528\u65f6')
$taskLiveWidth = 0
$taskCurrentDate = ''
$taskCurrentIndex = ''
$taskDayTimer = [System.Diagnostics.Stopwatch]::new()

function Write-TaskLiveLine([string]$Text) {
    $taskPad = ' ' * [Math]::Max(0, $script:taskLiveWidth - $Text.Length)
    Write-Host -NoNewline ("`r" + $Text + $taskPad)
    $script:taskLiveWidth = $Text.Length
}

function Write-TaskFinalLine([string]$Text) {
    Write-TaskLiveLine $Text
    Write-Host ''
    $script:taskLiveWidth = 0
}
$taskTimer = [System.Diagnostics.Stopwatch]::StartNew()
try {
    # Native stderr includes ordinary Python warnings; keep them in the full
    # log without making PowerShell abort an otherwise healthy forecast.
    $ErrorActionPreference = 'Continue'
    & $taskPython -u -m experiment_2025 --start-date 2025-01-01 --end-date 2025-12-31 --experiments $Experiment --lead-hours $taskLeads --output-root $taskOutputRoot 2>&1 | ForEach-Object {
        $taskLine = $_.ToString()
        Add-Content -LiteralPath $taskLogPath -Value $taskLine -Encoding UTF8
        if ($taskLine -match $taskStartPattern) {
            $taskCurrentIndex = $Matches['index']
            $taskCurrentDate = $Matches['date']
            $taskDayTimer.Restart()
            Write-TaskLiveLine "$taskRunningTag $taskCurrentIndex $taskCurrentDate"
        } elseif ($taskLine -match $taskRunningPattern) {
            Write-TaskLiveLine "$taskRunningTag $taskCurrentIndex $taskCurrentDate $taskElapsedTag $($taskDayTimer.Elapsed.ToString('hh\:mm\:ss'))"
        } elseif ($taskLine -match $taskEndPattern) {
            $taskDayTimer.Stop()
            Write-TaskFinalLine "$taskCompleteTag $taskCurrentIndex $taskCurrentDate $taskElapsedTag $($taskDayTimer.Elapsed.ToString('hh\:mm\:ss'))"
        } elseif ($taskLine -match $taskErrorPattern) {
            $taskDayTimer.Stop()
            Write-TaskFinalLine $taskLine
        }
    }
    $taskExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($taskExitCode -eq 0) {
        Write-Host 'Completed: all 365 dates passed validation.' -ForegroundColor Green
    } else {
        Write-Host "Ended with exit code $taskExitCode; see the detailed log and _runs JSON report." -ForegroundColor Yellow
    }
} finally {
    if ($taskLiveWidth -gt 0) { Write-Host '' }
    $taskTimer.Stop()
    Write-Host "Elapsed: $($taskTimer.Elapsed.ToString())"
    Write-Host "Details: $taskLogPath"
}

