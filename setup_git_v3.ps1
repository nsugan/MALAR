# MALAR — V3 git setup. Run from the project root in PowerShell.
# PAUSE OneDrive sync first (OneDrive locks .git and corrupts it).
$ErrorActionPreference = "Stop"
Write-Host "Setting up Git for MALAR V3..." -ForegroundColor Cyan
if (Test-Path .git) { Remove-Item -Recurse -Force .git }
git init | Out-Null
git add -A
git commit -m "MALAR v2 - full as-built (engine M0-M12 + web UI U0-U8 + LLM layer)" | Out-Null
git tag -a v2 -m "MALAR V2 - frozen as-built backup"
git branch v3
git checkout v3
git commit --allow-empty -m "Open V3 development line (3.0.0.dev0)" | Out-Null
Write-Host "Done. Branches/tags:" -ForegroundColor Green
git log --oneline --decorate -n 5
git tag
Write-Host "`nYou are now on branch 'v3'. Tag 'v2' preserves the frozen backup." -ForegroundColor Green
