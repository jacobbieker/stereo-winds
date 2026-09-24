# Global geostationary ring student AMV inference
# (scripts/infer_student_global_ring.py), configured through AMV_* env vars.
# See docker/entrypoint.sh for the full list.
#
#   docker build -t stereo-winds .
#   docker run --gpus all -e AMV_TIME=2025-03-10T12:00 \
#       -v "$PWD/output:/output" stereo-winds

ARG PIXI_VERSION=0.81.0

FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION} AS build
WORKDIR /app
# Dependencies first so source edits do not invalidate the solved env.
COPY pyproject.toml pixi.lock README.md LICENSE NOTICE ./
COPY stereo_winds/ stereo_winds/
COPY operational/ operational/
RUN pixi install --locked -e default \
 && pixi shell-hook -e default -s bash > /activate.sh \
 && rm -rf /root/.cache

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Same /app prefix as the build stage: the env's prefix paths and the
# editable stereo-winds install both point here.
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build /activate.sh /activate.sh
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY stereo_winds/ stereo_winds/
COPY operational/ operational/
COPY scripts/infer_student_global_ring.py scripts/
COPY checkpoints/student.abi.mb-v3.ep21.ckpt checkpoints/windflow.raft.sonde-tuned.ckpt checkpoints/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
# Convert the RAFT checkpoint now: StereoDisparity otherwise writes the
# _compat.pt beside it on first use, which fails for a non-root user.
RUN chmod +x /usr/local/bin/entrypoint.sh \
 && bash -c 'source /activate.sh && python -c "from stereo_winds.disparity import _ensure_compat_checkpoint as f; f(\"/app/checkpoints/windflow.raft.sonde-tuned.ckpt\")"'

ENV AMV_OUTPUT_DIR=/output \
    AMV_DEVICE=cuda \
    AMV_STUDENT_CKPT=/app/checkpoints/student.abi.mb-v3.ep21.ckpt \
    AMV_RAFT_CKPT=/app/checkpoints/windflow.raft.sonde-tuned.ckpt \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg
VOLUME /output

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
