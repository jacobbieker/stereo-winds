# openclimatefix/satellite-consumer, pinned to the commit we tested.
#
# Built here rather than from the repo's own Dockerfile, which cannot
# read the EUMETSAT products at all.  That Dockerfile installs with
# `uv sync` from uv.lock, but netcdf4, h5netcdf and pyspectral are
# declared under [tool.pixi.dependencies] and appear nowhere in uv.lock,
# so satpy ends up with no fci_l1c_nc or seviri_l1b_native reader and a
# run downloads every file for the window before failing with
#
#     cannot find module 'satpy.readers.fci_l1c_nc' (No module named 'netCDF4')
#
# Installing those three from PyPI instead is not enough: imports then
# succeed and the read fails deeper, inside dask, with
#
#     RuntimeError: NetCDF: HDF error
#
# because the wheels do not agree with each other about HDF5.  The conda
# builds are what the project actually declares, and on the same MTG
# window they process it cleanly (errs=0, a 366 MB store), so this image
# installs with pixi.
#
#   docker build -f docker/satellite-consumer.Dockerfile -t satellite-consumer .
#
# The build context supplies only the lock file; the source comes from
# the pinned checkout.

ARG PIXI_VERSION=0.70.2

FROM python:3.12-slim-bookworm AS source

ARG SATCONS_REPO=https://github.com/openclimatefix/satellite-consumer.git
# openclimatefix/satellite-consumer#44, "Change to return Datasets, add
# support for GOES/Himawari/GK-2A satellites".  Pinned by SHA, not by
# branch: the PR says it is not intended to be merged, and a force-push
# would otherwise change what this image is without changing this file.
ARG SATCONS_COMMIT=80c1b400317406f2c06491b7aa78949a98a1924c

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# Fetch the one commit rather than the branch.  .git stays: the package
# version comes from setuptools-git-versioning.
RUN git init -q . \
 && git remote add origin "${SATCONS_REPO}" \
 && git fetch -q --depth 1 origin "${SATCONS_COMMIT}" \
 && git checkout -q FETCH_HEAD

FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION} AS build
WORKDIR /opt/app
COPY --from=source /src/ ./
# The upstream commit ships no pixi.lock, so pixi would solve afresh on
# every build and the image would drift without this file changing.  The
# vendored lock was generated from this exact commit; --locked fails the
# build if the two ever disagree.
COPY docker/satellite-consumer.pixi.lock ./pixi.lock
RUN pixi install --locked -e default \
 && pixi shell-hook -e default -s bash > /activate.sh \
 && rm -rf /root/.cache

# Fail the build, not a 3am run, if a reader is missing again.
COPY docker/satellite-consumer-selfcheck.py /tmp/selfcheck.py
RUN bash -c 'source /activate.sh && python /tmp/selfcheck.py'

FROM ubuntu:24.04 AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# Same /opt/app prefix as the build stage: the environment's prefix paths
# and the editable satellite-consumer install both point here.
WORKDIR /opt/app
COPY --from=build /opt/app /opt/app
COPY --from=build /activate.sh /activate.sh

ENV SATCONS_WORKDIR=/work \
    PYTHONUNBUFFERED=1
VOLUME /work

# The console script lives in the pixi environment, which needs its
# activation script sourced for the prefix's shared libraries.
ENTRYPOINT ["/bin/bash", "-c", "source /activate.sh && exec sat-consumer \"$@\"", "--"]
