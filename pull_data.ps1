# Kéo dữ liệu về project (Windows PowerShell).
# Chạy từ thư mục project:
#   powershell -ExecutionPolicy Bypass -File .\pull_data.ps1
#   powershell -ExecutionPolicy Bypass -File .\pull_data.ps1 -n 500 -split train
#   powershell -ExecutionPolicy Bypass -File .\pull_data.ps1 -SkipDownload

param(
    [string]$Dataset = "yerevann/coco-karpathy",
    [string]$Split = "test",
    [int]$N = 300,
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$argsList = @("--dataset", $Dataset, "--split", $Split, "--n", $N)
if ($SkipDownload) { $argsList += "--skip-download" }

python pull_data.py @argsList
if ($LASTEXITCODE -ne 0) {
    Write-Error "pull_data.py thất bại (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}
