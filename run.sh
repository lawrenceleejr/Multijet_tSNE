#!/usr/bin/env bash
# run.sh — Docker wrapper for multijet_tsne.py
#
# Build the image once:
#   docker build -t multijet-tsne .
#
# Then run:
#   ./run.sh <hdf5-file-or-directory> [options]
#
# Examples:
#   ./run.sh ./data/events.h5
#   ./run.sh ./data/                          # all *.h5 files in directory
#   ./run.sh ./data/ --max-events 2000
#   ./run.sh ./data/ --algo tsne --perplexity 50
#
# All output files land in ./output/  (created automatically):
#   multijet_embedding.png   — embedding scatter plot
#   embedding_slices.png     — per-feature slice plots
#   umap_model.joblib        — fitted UMAP model for transform() on new data

set -euo pipefail

IMAGE_NAME="multijet-tsne"

# ─── Check docker is available ──────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "[error] Docker is not installed or not in PATH." >&2
    exit 1
fi

# ─── Require at least one argument ──────────────────────────────────────────
if [ $# -eq 0 ]; then
    cat >&2 <<EOF
Usage: ./run.sh <hdf5-file-or-directory> [options]

  <hdf5-file-or-directory>   Path to an HDF5 file or a directory of *.h5 files

Options (passed to multijet_tsne.py):
  --max-events N       Maximum events to process
  --algo {tsne,umap}   Embedding algorithm (default: umap)
  --n-neighbors N      UMAP n_neighbors (default: 15)
  --min-dist F         UMAP min_dist    (default: 0.1)
  --perplexity F       t-SNE perplexity (default: 30)
  --no-normalize       Skip StandardScaler
  --seed INT           Random seed (default: 42)
  --n-iter INT         t-SNE iterations (default: 1000)
  --verbose            Verbose per-file progress

Build the Docker image first if you haven't already:
  docker build -t ${IMAGE_NAME} .
EOF
    exit 1
fi

INPUT="$1"
shift

# Resolve to absolute path
INPUT="$(realpath "$INPUT")"

# ─── Check image exists ──────────────────────────────────────────────────────
if ! docker image inspect "${IMAGE_NAME}" &>/dev/null; then
    echo "[info] Image '${IMAGE_NAME}' not found — building now..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"
fi

# ─── Mount strategy ──────────────────────────────────────────────────────────
# If the user passed a directory, mount it directly as /data and pass /data.
# If the user passed a file, mount its parent as /data and translate the path.
if [ -d "$INPUT" ]; then
    DATA_MOUNT="$INPUT"
    CONTAINER_ARG="/data"
elif [ -f "$INPUT" ]; then
    DATA_MOUNT="$(dirname "$INPUT")"
    CONTAINER_ARG="/data/$(basename "$INPUT")"
else
    echo "[error] Input path does not exist: $INPUT" >&2
    exit 1
fi

# ─── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR="$(pwd)/output"
mkdir -p "$OUTPUT_DIR"

# ─── Run ─────────────────────────────────────────────────────────────────────
echo "[info] Data mount:  $DATA_MOUNT → /data"
echo "[info] Output dir:  $OUTPUT_DIR → /output"
echo ""

docker run --rm \
    -v "${DATA_MOUNT}:/data:ro" \
    -v "${OUTPUT_DIR}:/output" \
    "${IMAGE_NAME}" \
    "${CONTAINER_ARG}" \
    --output      /output/multijet_embedding.png \
    --slice-plot  /output/embedding_slices.png \
    --umap-output /output/umap_model.joblib \
    "$@"

echo ""
echo "[done] Output written to: ${OUTPUT_DIR}/"
