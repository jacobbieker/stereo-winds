# Operational AMV pipeline

Dagster orchestration around the research code in `stereo_winds/`. It turns
the single-satellite ("student") AMV retrieval into a service: watch for new
geostationary imagery, retrieve winds for each satellite independently, mosaic
the disks onto one global grid, and publish that mosaic to an
[icechunk](https://icechunk.io) store.

Nothing here reimplements science. Retrieval, mosaicking and store writing all
call upstream `stereo_winds` code; this package only schedules it, decides what
is runnable, and records what succeeded.

## What runs, and why it is split this way

Each operational timestamp (default: every hour, see `cadence_minutes`) moves
through four stages:

| Stage | Role |
| --- | --- |
| **Availability** | Ask each satellite's archive whether imagery for this slot exists yet, within `availability_tolerance_minutes` of the nominal time. Produces the list of satellites that are runnable now. |
| **Per-satellite AMV retrieval** | One step **per satellite** (GOES-18/19, Himawari-9, GK-2A). Runs the student model over that satellite's full disk and writes a per-satellite AMV dataset — the 7 variables `u_wind`, `v_wind`, `cloud_top_height`, `quality_flag`, `sigma_u`, `sigma_v`, `sigma_h` on the satellite's own `(y, x)` grid, with 2-D `latitude` / `longitude` / `zenith_angle`. |
| **Mosaic** | Merge the per-satellite disks onto a regular lat/lon grid at `resolution_m`, keeping the lowest-satellite-zenith contributor in each cell. |
| **Publish** | Append the mosaic to an icechunk store along the `time` dimension, and commit. |

The retrieval stage is **one step per satellite, on purpose**. A GK-2A download
that times out should not cost you the three disks that already ran: the failed
satellite alone is re-materialized, and the mosaic then picks up every
per-satellite output that exists. Dagster models this as a satellite-partitioned
asset, so "resume GK-2A for 12:00 UTC" is a single-partition backfill rather
than a re-run of the whole timestamp.

## Asset graph

```
               availability (per timestamp)
                          │
          ┌───────────┬───┴───────┬────────────┐
          ▼           ▼           ▼            ▼
      goes18       goes19     himawari9      gk2a      ← per-satellite AMVs
          └───────────┴─────┬─────┴────────────┘
                            ▼
                     global mosaic
                            ▼
                icechunk store (local or S3)
```

All Dagster commands below target `operational.definitions`, the module that
exports the `Definitions` object. The exact asset keys are whatever
`dagster asset list` prints — use that as the source of truth when scripting:

```bash
export PYTHONPATH=$PWD
dagster asset list -m operational.definitions
```

## Install

Dagster is an optional extra:

```bash
pip install -e ".[operational]"
# or, with pixi
pixi install -e operational
```

## Run it

Launch the Dagster UI and scheduler locally:

```bash
export PYTHONPATH=$PWD
dagster dev -m operational.definitions
```

Then open <http://localhost:3000>. From the **Assets** tab you can materialize
the graph for a timestamp, inspect per-satellite run logs, and see which
satellites were reported available.

Headless, for a single timestamp:

```bash
dagster asset materialize -m operational.definitions \
    --select '*' --partition 2024-01-15-12:00
```

## Backfill a time range

Backfills run the same assets over many timestamp partitions. From the UI:
**Assets → select the graph → Materialize → choose a partition range**. From the
CLI:

```bash
dagster asset backfill -m operational.definitions \
    --select '*' \
    --partition-range 2024-01-15-00:00...2024-01-16-00:00
```

Partitions are cadence-aligned (`OperationalConfig.cadence_minutes`), so a
range given in wall-clock terms is snapped to the operational slots —
`OperationalConfig.timestamps(start, end)` enumerates exactly the slots a
backfill will cover.

Backfills are throughput-bound on inference; keep concurrency at or below the
number of GPUs by setting a run-queue limit in `dagster.yaml`, or run with
`device="cpu"` only for smoke tests.

## Resume a single failed satellite

When one satellite fails, everything else for that timestamp is already
materialized. Re-run only the failure:

```bash
# Re-run GK-2A for the 12:00 UTC slot, then the downstream mosaic + publish.
dagster asset materialize -m operational.definitions \
    --select 'satellite_amv+' \
    --partition 'gk2a|2024-01-15-12:00'
```

In the UI the equivalent is: open the per-satellite asset, pick the failed
satellite partition, **Materialize selected**, and tick *materialize downstream
assets*.

The mosaic step tolerates a missing satellite — it merges whichever disks are
present and records the contributors in the output attributes — so a satellite
that is permanently unavailable for a slot does not block publication. Check
`source_satellite` / the contributor attributes on the published mosaic to see
what actually went in.

## Where the output goes

`OperationalConfig.store_uri` selects the icechunk destination, and the publish
step branches on it:

* **Local path** — anything without a scheme, e.g.

  ```bash
  export STEREO_WINDS_OP_STORE_URI=/data/amv/global.icechunk
  ```

  The store is created on first write under that directory. Good for
  development and for the integration tests, which point it at `tmp_path`.

* **S3** — an `s3://bucket/prefix` URI, e.g.

  ```bash
  export STEREO_WINDS_OP_STORE_URI=s3://my-bucket/amv/global
  ```

  Credentials come from the usual AWS chain (`AWS_PROFILE`, instance role,
  `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`); set `AWS_REGION` for the
  bucket's region. `OperationalConfig.store_bucket_prefix()` splits the URI into
  the `(bucket, prefix)` pair the icechunk S3 storage config wants.

Switching between the two changes nothing else in the graph — the same publish
step handles both.

## Configuration

Everything is one frozen dataclass, `operational.config.OperationalConfig`:

| Field | Default | Meaning |
| --- | --- | --- |
| `satellites` | `("goes18", "goes19", "himawari9", "gk2a")` | Ring to retrieve, in mosaic order |
| `flow_bands` | `DEFAULT_FLOW_BANDS` | Bands optical flow runs on |
| `rad_bands` | `DEFAULT_RAD_BANDS` | Brightness-temperature bands |
| `cadence_minutes` | `60` | Spacing of operational slots |
| `availability_tolerance_minutes` | `5.0` | Slack between a scene's time and its slot |
| `output_dir` | `output/operational` | Intermediate per-satellite and mosaic files |
| `store_uri` | `output/operational.icechunk` | Publish destination (local path or `s3://`) |
| `resolution_m` | `10000.0` | Mosaic grid spacing |
| `device` | `"cpu"` | Torch device for inference |
| `row_strip` | `1024` | Full-disk rows per forward-pass strip |

Every field can be set from the environment with the `STEREO_WINDS_OP_` prefix
and the upper-cased field name; sequences are comma-separated:

```bash
export STEREO_WINDS_OP_SATELLITES="goes19,himawari9"
export STEREO_WINDS_OP_CADENCE_MINUTES=10
export STEREO_WINDS_OP_DEVICE=cuda
export STEREO_WINDS_OP_STORE_URI=s3://my-bucket/amv/global
python -c "from operational.config import OperationalConfig; print(OperationalConfig.from_env())"
```

Model weights and data locations still come from the upstream environment
variables (`STEREO_WINDS_RAFT_CKPT`, `STEREO_WINDS_DATA_DIR`).

## Tests

```bash
export PYTHONPATH=$PWD
python -m pytest operational/tests -q
```

The suite is fully synthetic — no network, no GPU, no checkpoints.
`operational/tests/conftest.py` provides `synthetic_scene(sat_id, t0, ...)`,
which builds a per-satellite AMV dataset with the same schema as the real
retrieval, plus the `op_config` and `tmp_store_uri` fixtures that keep all
writes under `tmp_path`.
