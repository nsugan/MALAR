#!/usr/bin/env bash
# MALAR — full project backup (Linux/WSL). Captures folder + Docker volumes + images.
#   ./backup_project.sh -d /mnt/e/MALAR_backup [--skip-images] [--skip-ollama] [--no-stop]
set -euo pipefail

# ensure the alpine helper (used to archive/extract volumes) is present
if ! docker image inspect alpine:latest >/dev/null 2>&1; then
  echo "==> pulling alpine helper image..."; docker pull alpine:latest >/dev/null || { echo "ERROR: cannot pull alpine (needed for volume archiving)"; exit 1; }
fi

DEST="/mnt/backup/MALAR_backup"
SKIP_IMAGES=0; SKIP_OLLAMA=0; NO_STOP=0
while [ $# -gt 0 ]; do
  case "$1" in
    -d|--dest) DEST="$2"; shift 2;;
    --skip-images) SKIP_IMAGES=1; shift;;
    --skip-ollama) SKIP_OLLAMA=1; shift;;
    --no-stop) NO_STOP=1; shift;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="$DEST/MALAR_$STAMP"
mkdir -p "$ROOT/volumes" "$ROOT/images"
echo "==> Backup target: $ROOT"

[ "$NO_STOP" -eq 0 ] && { echo "==> Stopping stack for a consistent snapshot..."; docker compose stop >/dev/null; }

# 1) project folder (keep data/, .env, infra/; drop regenerable heavy dirs)
echo "==> Copying project folder (source + config + data)..."
rsync -a --exclude node_modules --exclude .venv --exclude .git \
      --exclude '__pycache__' --exclude '*.pyc' ./ "$ROOT/project/"

# 2) Docker named volumes -> tar.gz (auto-detect by suffix, any project prefix)
SUFFIXES=(neo4j_data qdrant_data agent_db)
[ "$SKIP_OLLAMA" -eq 0 ] && SUFFIXES+=(ollama_models)
VOLS="$(docker volume ls --format '{{.Name}}')"
for sfx in "${SUFFIXES[@]}"; do
  vol="$(echo "$VOLS" | grep -E "(_${sfx}\$|^${sfx}\$)" | head -1 || true)"
  if [ -n "$vol" ]; then
    echo "==> Volume $vol -> $sfx.tar.gz"
    docker run --rm -v "${vol}:/data:ro" -v "${ROOT}/volumes:/backup" \
      alpine sh -c "tar czf /backup/${sfx}.tar.gz -C /data ." 2>/dev/null
    echo "$sfx=$vol" >> "$ROOT/volumes/_volume_map.txt"
  else
    echo "!! volume *_$sfx not found (skipped)"
  fi
done

# 3) Docker images used by the project
if [ "$SKIP_IMAGES" -eq 0 ]; then
  echo "==> Saving docker images (large; use --skip-images to skip)..."
  IMGS="$(docker compose config --images 2>/dev/null | sort -u || true)"
  [ -z "$IMGS" ] && IMGS=$'neo4j:5-community\nqdrant/qdrant:latest\nollama/ollama:latest\nghcr.io/berriai/litellm:main-stable'
  echo "$IMGS" > "$ROOT/images/_image_list.txt"
  # shellcheck disable=SC2086
  docker save -o "$ROOT/images/images.tar" $IMGS
fi

# 4) manifest + restore note
docker volume ls > "$ROOT/manifest_volumes.txt" 2>/dev/null || true
docker compose images > "$ROOT/manifest_images.txt" 2>/dev/null || true
cat > "$ROOT/README_RESTORE.txt" <<EOF
MALAR backup $STAMP
  project/           source + config + .env + data/
  volumes/*.tar.gz   Neo4j, Qdrant, Ollama model, agent SQLite (the LEARNED state)
  images/images.tar  docker images (optional; can be rebuilt/re-pulled)
Restore:  ./restore_project.sh -s "$ROOT" -i ~/MALAR
EOF

[ "$NO_STOP" -eq 0 ] && docker compose start >/dev/null
echo "==> DONE. Backup at $ROOT ($(du -sh "$ROOT" | cut -f1))"
