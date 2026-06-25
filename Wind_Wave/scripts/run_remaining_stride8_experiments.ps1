param(
    [string]$Python = "C:\Users\SLDUO\anaconda3\envs\pangu\python.exe"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$env:PYTHONUNBUFFERED = "1"

$experiments = @(
    [ordered]@{
        Name = "M2wave0c72__in_past24hWind+future72hWind+waveT0__direct_nores__s8_b1024_h64_bf16__ep20_pause40__y2016-2025"
        Variant = "m2-wave0-direct"
        FutureWindMode = "continuous72"
    },
    [ordered]@{
        Name = "M2wave0__in_past24hWind+futureLeadWind+waveT0__residual__s8_b1024_h64_bf16__ep20_pause40__y2016-2025"
        Variant = "m2-wave0-residual"
        FutureWindMode = "target"
    },
    [ordered]@{
        Name = "M2wave0c72__in_past24hWind+future72hWind+waveT0__residual__s8_b1024_h64_bf16__ep20_pause40__y2016-2025"
        Variant = "m2-wave0-residual"
        FutureWindMode = "continuous72"
    }
)

function Get-TrainLogRows {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return -1
    }
    return [Math]::Max(0, ((Get-Content -LiteralPath $Path | Measure-Object -Line).Lines - 1))
}

Write-Output "runner_start=$(Get-Date -Format o)"
Write-Output "project_root=$ProjectRoot"

foreach ($experiment in $experiments) {
    $runName = $experiment.Name
    $outputDir = Join-Path $ProjectRoot "outputs\$runName"
    $trainLog = Join-Path $outputDir "logs\train_log.csv"
    $rows = Get-TrainLogRows -Path $trainLog

    if ($rows -ge 20) {
        Write-Output "skip_complete run=$runName train_log_rows=$rows"
        continue
    }

    if ((Test-Path -LiteralPath $outputDir) -and $rows -ne -1) {
        throw "Refusing to overwrite incomplete run: $runName train_log_rows=$rows"
    }

    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    $consoleLog = Join-Path $outputDir "console.log"

    $arguments = @(
        "-m", "wind_wave.train",
        "--data-source", "zarr",
        "--zarr-dir", "data\zarr",
        "--metadata-dir", "data\metadata",
        "--years", "2016-2025",
        "--epochs", "20",
        "--batch-size", "1024",
        "--weight-decay", "1e-4",
        "--dropout", "0.1",
        "--precision", "bf16",
        "--hidden-channels", "64",
        "--spatial-stride", "8",
        "--model-variant", $experiment.Variant,
        "--future-wind-mode", $experiment.FutureWindMode,
        "--run-name", $runName,
        "--fast-in-memory-dataset",
        "--pin-memory",
        "--epoch-pause-seconds", "40",
        "--device", "cuda"
    )

    Write-Output "run_start=$(Get-Date -Format o) run=$runName variant=$($experiment.Variant) future_wind_mode=$($experiment.FutureWindMode)"
    Write-Output "console_log=$consoleLog"
    & $Python @arguments 2>&1 | Tee-Object -FilePath $consoleLog
    $exitCode = $LASTEXITCODE
    Write-Output "run_end=$(Get-Date -Format o) run=$runName exit_code=$exitCode"

    if ($exitCode -ne 0) {
        throw "Training failed for $runName with exit code $exitCode"
    }
}

Write-Output "runner_end=$(Get-Date -Format o)"
