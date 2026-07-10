<#  MALAR — restore a backup made by backup_project.ps1.
    powershell -ExecutionPolicy Bypass -File .\restore_project.ps1 -Src D:\MALAR_backup\MALAR_YYYYMMDD_HHMMSS -Into C:\Dev\MALAR
#>
param(
  [Parameter(Mandatory=$true)][string]$Src,
  [string]$Into = "C:\Dev\MALAR"
)
$ErrorActionPreference = "Continue"
function Die($m){ Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Src)) { Die "backup folder not found: $Src" }

docker info *> $null
if ($LASTEXITCODE -ne 0) { Die "Docker Desktop isn't running. Start it, wait for 'Engine running', then retry." }
docker image inspect alpine:latest *> $null
if ($LASTEXITCODE -ne 0) { Write-Host "==> pulling alpine helper..."; docker pull alpine:latest *> $null }

Write-Host "==> Restoring project folder -> $Into"
New-Item -ItemType Directory -Force -Path $Into | Out-Null
robocopy "$Src\project" $Into /E /R:1 /W:1 /NFL /NDL /NJH /NJS | Out-Null
if ($LASTEXITCODE -ge 8) { Die "robocopy failed ($LASTEXITCODE)" }
Set-Location $Into

if (Test-Path "$Src\images\images.tar") { Write-Host "==> Loading docker images..."; docker load -i "$Src\images\images.tar" }

Write-Host "==> Creating containers/volumes..."
docker compose create *> $null
$vols = docker volume ls --format "{{.Name}}"
Get-ChildItem "$Src\volumes\*.tar.gz" | ForEach-Object {
  $sfx = $_.Name -replace '\.tar\.gz$',''
  $vol = $vols | Where-Object { $_ -match "_$sfx$" -or $_ -eq $sfx } | Select-Object -First 1
  if ($vol) {
    Write-Host "==> Importing $($_.Name) -> $vol"
    docker run --rm -v "${vol}:/data" -v "$($_.DirectoryName):/backup:ro" alpine sh -c "cd /data && tar xzf /backup/$($_.Name)" 2>&1 | Out-Null
  } else { Write-Host "   !! target volume for $sfx not found; run 'docker compose up -d' once then re-run restore" -ForegroundColor Yellow }
}

Write-Host "==> Starting the stack..."
docker compose up -d --build
Write-Host "==> DONE. Open http://localhost:3000 (re-pull the model only if ollama_models was skipped)." -ForegroundColor Green
