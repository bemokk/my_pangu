param()
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$taskPython = 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe'
$taskOutputRoot = 'H:\pangu_2025_data\model_output\experiment_2025'
$taskLogDir = Join-Path $taskOutputRoot '_runs'
$taskRunStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$taskLogPath = Join-Path $taskLogDir "gdas_lagged_year_detail_$taskRunStamp.log"
$taskLeads = (0..40 | ForEach-Object { $_ * 3 }) -join ','
Set-Location -LiteralPath $taskProjectRoot
New-Item -ItemType Directory -Path $taskLogDir -Force | Out-Null
$Host.UI.RawUI.WindowTitle = 'gdas_lagged_120h 2025 | 365 days | 0-120h / 3h'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $OutputEncoding
Write-Host 'gdas_lagged_120h | 2025-01-01 to 2025-12-31 | 00 UTC' -ForegroundColor Cyan
Write-Host '365 dates x 41 leads (0-120h / 3h). GDAS initial time: T0-120h.'
Write-Host 'Valid existing results are reused.'
Write-Host "Details: $taskLogPath"
Write-Host 'Ctrl+C: interrupt; run this script again to resume.'
Write-Host ''
$taskStartPattern = '^\[\u4efb\u52a1\u5f00\u59cb\]\s+(?<index>\d+/\d+)\s+\S+\s+(?<date>\S+)'
$taskRunningPattern = '^\[\u8fd0\u884c\u4e2d\]'
$taskBatchPattern = '^\[\u6279\u6b21\u8fdb\u5ea6\]'
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
    $ErrorActionPreference = 'Continue'
    & $taskPython -u -m experiment_2025 --start-date 2025-01-01 --end-date 2025-12-31 --experiments gdas_lagged_120h --lead-hours $taskLeads --output-root $taskOutputRoot 2>&1 | ForEach-Object {
        $taskLine = $_.ToString()
        Add-Content -LiteralPath $taskLogPath -Value $taskLine -Encoding UTF8
        if ($taskLine -match $taskStartPattern) {
            $taskCurrentIndex = $Matches['index']
            $taskCurrentDate = $Matches['date']
            $taskDayTimer.Restart()
            Write-TaskLiveLine "$taskRunningTag $taskCurrentIndex $taskCurrentDate"
        } elseif ($taskLine -match $taskRunningPattern) {
            Write-TaskLiveLine "$taskRunningTag $taskCurrentIndex $taskCurrentDate $taskElapsedTag $($taskDayTimer.Elapsed.ToString('hh\:mm\:ss'))"
        } elseif ($taskLine -match $taskBatchPattern) {
            $taskDayTimer.Stop()
            Write-TaskFinalLine "$taskCompleteTag $taskCurrentIndex $taskCurrentDate $taskElapsedTag $($taskDayTimer.Elapsed.ToString('hh\:mm\:ss'))"
        }
    }
    $taskExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($taskExitCode -eq 0) { Write-Host 'Completed: all 365 dates passed validation.' -ForegroundColor Green }
    else { Write-Host "Ended with exit code $taskExitCode; see the detailed log and _runs JSON report." -ForegroundColor Yellow }
} finally {
    if ($taskLiveWidth -gt 0) { Write-Host '' }
    $taskTimer.Stop()
    Write-Host "Elapsed: $($taskTimer.Elapsed.ToString())"
    Write-Host "Details: $taskLogPath"
}


