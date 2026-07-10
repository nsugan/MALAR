<#  MALAR — full project backup to an external disk (folder + Docker volumes + images).
    Run from the project root:
      powershell -ExecutionPolicy Bypass -File .\backup_project.ps1 -Dest D:\MALAR_backup
#>
param(
  [string]$Dest = "D:\MALAR_backup",
  [switch]$SkipImages,   # skip docker image save (biggest part; images can be rebuilt/re-pulled)
  [switch]$SkipOllama,   # skip the ~5 GB ollama model volume
  [switch]$NoStop        # don't stop the stack (hot copy)
)
# NOTE: we do NOT use -ErrorActionPreference Stop, because native docker commands write
# informational lines (e.g. "Unable to find image locally") to stderr which would abort us.
$ErrorActionPreference = "Continue"
function Die($m){ Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

# 0) Docker must be running
docker info *> $null
if ($LASTEXITCODE -ne 0) { Die "Docker Desktop isn't running. Start it, wait until it says 'Engine running', then retry." }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$root  = Join-Path $Dest "MALAR_$stamp"
New-Item -ItemType Directory -Force -Path $root,"$root\volumes","$root\images" | Out-Null
Write-Host "==> Backup target: $root"

# 0b) make sure the tiny alpine helper (used to tar volumes) is present locally
docker image inspect alpine:latest *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Host "==> pulling alpine helper image (5 MB)..."
  docker pull alpine:latest *> $null
  if ($LASTEXITCODE -ne 0) { Die "could not pull 'alpine' (needed to archive the volumes). Check your internet/Docker Hub access and retry." }
}

if (-not $NoStop) { Write-Host "==> Stopping stack for a consistent snapshot..."; docker compose stop *> $null }

# 1) project folder (keep data/, .env, infra/; drop regenerable heavy dirs)
Write-Host "==> Copying project folder (source + config + data)..."
robocopy . "$root\project" /E /XD node_modules .venv .git "__pycache__" /XF "*.pyc" /R:1 /W:1 /NFL /NDL /NJH /NJS | Out-Null
if ($LASTEXITCODE -ge 8) { Die "robocopy failed ($LASTEXITCODE)" }

# 2) Docker named volumes -> tar.gz (auto-detect by suffix, any project prefix)
$suffixes = @("neo4j_data","qdrant_data","agent_db")
if (-not $SkipOllama) { $suffixes += "ollama_models" }
$vols = docker volume ls --format "{{.Name}}"
foreach ($sfx in $suffixes) {
  $vol = $vols | Where-Object { $_ -match "_$sfx$" -or $_ -eq $sfx } | Select-Object -First 1
  if ($vol) {
    Write-Host "==> Volume $vol -> $sfx.tar.gz"
    docker run --rm -v "${vol}:/data:ro" -v "${root}\volumes:/backup" alpine sh -c "tar czf /backup/$sfx.tar.gz -C /data ." 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "   !! failed archiving $vol (skipped)" -ForegroundColor Yellow }
    else { "$sfx=$vol" | Add-Content "$root\volumes\_volume_map.txt" }
  } else { Write-Host "   !! volume *_$sfx not found (skipped)" -ForegroundColor Yellow }
}

# 3) Docker images used by the project
if (-not $SkipImages) {
  Write-Host "==> Saving docker images (large; use -SkipImages to skip)..."
  $imgs = (docker compose config --images 2>$null) | Sort-Object -Unique | Where-Object { $_ }
  if (-not $imgs) { $imgs = @("neo4j:5-community","qdrant/qdrant:latest","ollama/ollama:latest","ghcr.io/berriai/litellm:main-stable") }
  $imgs | Set-Content "$root\images\_image_list.txt"
  docker save -o "$root\images\images.tar" @imgs 2>&1 | Out-Null
  if ($LASTEXITCODE -ne 0) { Write-Host "   !! image save failed (continuing without images)" -ForegroundColor Yellow }
}

# 4) manifest + restore note
docker volume ls   | Out-File "$root\manifest_volumes.txt"
docker compose images 2>$null | Out-File "$root\manifest_images.txt"
@"
MALAR backup $stamp
  project\           source + config + .env + data\
  volumes\*.tar.gz   Neo4j, Qdrant, Ollama model, agent SQLite (the LEARNED state)
  images\images.tar  docker images (optional; can be rebuilt/re-pulled)
Restore with restore_project.ps1 (in project\).
"@ | Out-File "$root\README_RESTORE.txt"

if (-not $NoStop) { docker compose start *> $null }
try { $size = "{0:N1} GB" -f ((Get-ChildItem $root -Recurse | Measure-Object Length -Sum).Sum / 1GB) } catch { $size = "?" }
Write-Host "==> DONE. Backup at $root  (total $size)" -ForegroundColor Green
