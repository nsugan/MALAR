#!/usr/bin/env bash
# MALAR — restore a backup made by backup_project.sh (Linux/WSL).
#   ./restore_project.sh -s /mnt/e/MALAR_backup/MALAR_YYYYMMDD_HHMMSS -i ~/MALAR
set -euo pipefail

# ensure the alpine helper (used to archive/extract volumes) is present
if ! docker image inspect alpine:latest >/dev/null 2>&1; then
  echo "==> pulling alpine helper image..."; docker pull alpine:latest >/dev/null || { echo "ERROR: cannot pull alpine (needed for volume archiving)"; exit 1; }
fi

SRC=""; INTO="$HOME/MALAR"
while [ $# -gt 0 ]; do
  case "$1" in
    -s|--src) SRC="$2"; shift 2;;
    -i|--into) INTO="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done
[ -z "$SRC" ] && { echo "usage: $0 -s <backup_dir> [-i <install_dir>]"; exit 1; }
[ -d "$SRC" ] || { echo "backup folder not found: $SRC"; exit 1; }

echo "==> Restoring project folder -> $INTO"
mkdir -p "$INTO"
rsync -a "$SRC/project/" "$INTO/"
cd "$INTO"

if [ -f "$SRC/images/images.tar" ]; then
  echo "==> Loading docker images..."
  docker load -i "$SRC/images/images.tar"
fi

echo "==> Creating containers/volumes..."
docker compose create >/dev/null 2>&1 || true
VOLS="$(docker volume ls --format '{{.Name}}')"
for f in "$SRC"/volumes/*.tar.gz; do
  [ -e "$f" ] || continue
  base="$(basename "$f")"; sfx="${base%.tar.gz}"
  vol="$(echo "$VOLS" | grep -E "(_${sfx}\$|^${sfx}\$)" | head -1 || true)"
  if [ -n "$vol" ]; then
    echo "==> Importing $base -> $vol"
    docker run --rm -v "${vol}:/data" -v "$(dirname "$f"):/backup:ro" \
      alpine sh -c "cd /data && tar xzf /backup/$base" 2>/dev/null
  else
    echo "!! target volume for $sfx not found; run 'docker compose up -d' once, then re-run restore"
  fi
done

echo "==> Starting the stack..."
docker compose up -d --build
echo "==> DONE. Open http://localhost:3000 (re-pull the model only if ollama_models was skipped)."
