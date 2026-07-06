# Setting up Git for V3 (run on your machine)

I could not create the git repo from here: **Git does not work inside a OneDrive-synced
folder** — OneDrive locks `.git/config`/`.git/index`, which corrupts the repo
("Operation not permitted"). So do this yourself, OneDrive-safely.

## Recommended: keep the repo, but pause OneDrive while doing git operations
1. Right-click the OneDrive cloud icon → **Pause syncing → 2 hours**.
2. Open **PowerShell** in the project:
   ```powershell
   cd C:\Users\HP\OneDrive\MALAR_V2
   Remove-Item -Recurse -Force .git -ErrorAction SilentlyContinue   # clear any broken repo
   ```
3. Run the setup script (below), or the commands manually.
4. Resume OneDrive syncing.

## Better long-term: move the repo OUT of OneDrive
OneDrive + Git will keep fighting. Consider keeping the working repo at e.g.
`C:\dev\MALAR` and letting OneDrive hold only backups/zips. To move:
```powershell
robocopy C:\Users\HP\OneDrive\MALAR_V2 C:\dev\MALAR /E /XD node_modules .venv __pycache__ .git
cd C:\dev\MALAR
```
then run the setup below there.

## The setup (creates tag `v2`, branch `v3`)
```powershell
git init
git add -A
git commit -m "MALAR v2 - full as-built (engine M0-M12 + web UI U0-U8 + LLM layer)"
git tag -a v2 -m "MALAR V2 - frozen as-built backup"
git branch v3
git checkout v3
git commit --allow-empty -m "Open V3 development line (3.0.0.dev0)"
git log --oneline --decorate
```
`.gitignore` is already in the folder (excludes node_modules/.venv/caches/.env and the
stray test domains). `.env` is deliberately NOT committed — your API keys live only in the
local `.env` and in the zip backup.

## Optional: push to a private remote
```powershell
git remote add origin <your-private-repo-url>
git push -u origin v3
git push origin v2          # push the tag too
```
(Only push to a PRIVATE repo — even though `.env` is ignored, treat the history as sensitive.)
