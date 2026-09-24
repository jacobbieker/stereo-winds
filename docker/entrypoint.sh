#!/usr/bin/env bash
# Build the infer_student_global_ring.py command line from AMV_* env vars.
#
# Required:
#   AMV_TIME                        --time (ISO, e.g. 2025-03-10T12:00)
# Value options (unset or empty = script default):
#   AMV_END_TIME                    --end-time
#   AMV_STEP_MINUTES                --step-minutes
#   AMV_STUDENT_CKPT                --student-ckpt   (image default: bundled)
#   AMV_RAFT_CKPT                   --raft-ckpt      (image default: bundled)
#   AMV_OUTPUT_DIR                  --output-dir     (image default: /output)
#   AMV_DEVICE                      --device         (image default: cuda)
#   AMV_SATELLITES                  --satellites
#   AMV_FLOW_BANDS                  --flow-bands
#   AMV_RAD_BANDS                   --rad-bands
#   AMV_RESOLUTION_M                --resolution-m
#   AMV_ROW_STRIP                   --row-strip
#   AMV_AVAILABILITY_TOLERANCE      --availability-tolerance
#   AMV_SCENE_CACHE_GB              --scene-cache-gb
#   AMV_MEMORY_RESERVE_GB           --memory-reserve-gb
#   AMV_PREFETCH_WORKERS            --prefetch-workers
#   AMV_DOWNLOAD_WORKERS            --download-workers
#   AMV_ICECHUNK_STORE              --icechunk-store
#   AMV_ICECHUNK_BRANCH             --icechunk-branch
#   AMV_ICECHUNK_CHUNK              --icechunk-chunk
#   AMV_ICECHUNK_ENDPOINT           --icechunk-endpoint
#   AMV_ICECHUNK_REGION             --icechunk-region
# Flags (true/1/yes to enable):
#   AMV_SKIP_GLOBAL                 --skip-global
#   AMV_REQUIRE_ALL_SATELLITES      --require-all-satellites
#   AMV_SKIP_EXISTING               --skip-existing
#   AMV_NO_NETCDF                   --no-netcdf
#   AMV_ICECHUNK_ANONYMOUS          --icechunk-anonymous
#   AMV_ICECHUNK_FORCE_PATH_STYLE   --icechunk-force-path-style
# Anything else:
#   AMV_EXTRA_ARGS                  appended verbatim (word-split)
#   container arguments             appended after everything above
#
# S3 credentials for an icechunk store come from the usual AWS_* variables.
set -euo pipefail

# The image's activation script references unset variables.
set +u
# shellcheck disable=SC1091
source /activate.sh
set -u

if [[ -z "${AMV_TIME:-}" ]]; then
    echo "entrypoint: AMV_TIME is required (e.g. -e AMV_TIME=2025-03-10T12:00)" >&2
    exit 2
fi

args=()

opt() {  # opt ENV_VAR --flag
    local val="${!1:-}"
    if [[ -n "$val" ]]; then args+=("$2" "$val"); fi
}

flag() {  # flag ENV_VAR --flag
    local val="${!1:-}"
    case "${val,,}" in
        1|true|yes|on) args+=("$2") ;;
        ""|0|false|no|off) ;;
        *) echo "entrypoint: $1=$val is not a boolean" >&2; exit 2 ;;
    esac
}

opt AMV_TIME                   --time
opt AMV_END_TIME               --end-time
opt AMV_STEP_MINUTES           --step-minutes
opt AMV_STUDENT_CKPT           --student-ckpt
opt AMV_RAFT_CKPT              --raft-ckpt
opt AMV_OUTPUT_DIR             --output-dir
opt AMV_DEVICE                 --device
opt AMV_SATELLITES             --satellites
opt AMV_FLOW_BANDS             --flow-bands
opt AMV_RAD_BANDS              --rad-bands
opt AMV_RESOLUTION_M           --resolution-m
opt AMV_ROW_STRIP              --row-strip
opt AMV_AVAILABILITY_TOLERANCE --availability-tolerance
opt AMV_SCENE_CACHE_GB         --scene-cache-gb
opt AMV_MEMORY_RESERVE_GB      --memory-reserve-gb
opt AMV_PREFETCH_WORKERS       --prefetch-workers
opt AMV_DOWNLOAD_WORKERS       --download-workers
opt AMV_ICECHUNK_STORE         --icechunk-store
opt AMV_ICECHUNK_BRANCH        --icechunk-branch
opt AMV_ICECHUNK_CHUNK         --icechunk-chunk
opt AMV_ICECHUNK_ENDPOINT      --icechunk-endpoint
opt AMV_ICECHUNK_REGION        --icechunk-region

flag AMV_SKIP_GLOBAL               --skip-global
flag AMV_REQUIRE_ALL_SATELLITES    --require-all-satellites
flag AMV_SKIP_EXISTING             --skip-existing
flag AMV_NO_NETCDF                 --no-netcdf
flag AMV_ICECHUNK_ANONYMOUS        --icechunk-anonymous
flag AMV_ICECHUNK_FORCE_PATH_STYLE --icechunk-force-path-style

if [[ -n "${AMV_EXTRA_ARGS:-}" ]]; then
    read -r -a extra <<< "$AMV_EXTRA_ARGS"
    args+=("${extra[@]}")
fi

cmd=(python /app/scripts/infer_student_global_ring.py "${args[@]}" "$@")
echo "entrypoint: ${cmd[*]}" >&2
exec "${cmd[@]}"
