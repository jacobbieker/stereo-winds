"""Tests for the mosaic and publish assets.

Everything here is offline and synthetic, and the icechunk store is a real
local store under ``tmp_path`` — the publish path is the one place where a
mock would hide exactly the bugs worth catching (time encoding drift,
duplicate appends, vocabulary mismatches).
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
import xarray as xr
from dagster import materialize

from operational.assets.mosaic_assets import (
    amv_asset_name,
    build_mosaic_assets,
    global_mosaic,
    load_available_retrievals,
    per_satellite_path,
    published_mosaic,
)
from operational.core.partitions import key_for
from operational.core.publish import open_store
from operational.resources import (
    IcechunkStoreResource,
    PathsResource,
    RunSettingsResource,
)
from operational.tests.conftest import synthetic_scene

T0 = datetime(2024, 6, 1, 12, 0)
T1 = T0 + timedelta(hours=1)
SATELLITES = ["goes18", "goes19"]
# 200 km keeps the global grid at ~100 x 200 cells instead of the hundreds
# of millions an operational 10 km run would allocate.
TEST_RESOLUTION_M = 200_000.0


def seed_scene(
    output_dir: Path,
    sat_id: str,
    t0: datetime,
    zenith: float = 10.0,
    usable: bool = True,
) -> Path:
    """Write a synthetic per-satellite retrieval where the mosaic looks.

    ``usable=False`` writes a well-formed file whose pixels all fail the
    quality gate — the shape a bad inference run leaves behind.
    """
    path = per_satellite_path(output_dir, sat_id, t0)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = synthetic_scene(sat_id, t0, zenith=zenith)
    if not usable:
        scene["quality_flag"][:] = 0.0
    scene.to_netcdf(path)
    return path


def build_resources(output_dir: Path, store_uri: str, satellites=None) -> dict:
    """Resource set every test in this module runs against."""
    return {
        "paths": PathsResource(output_dir=str(output_dir)),
        "store": IcechunkStoreResource(store_uri=store_uri, branch="main", chunk=64),
        "run_settings": RunSettingsResource(
            satellites=list(SATELLITES if satellites is None else satellites),
            resolution_m=TEST_RESOLUTION_M,
            skip_existing=True,
        ),
    }


def materialization_metadata(result, node_name: str) -> dict:
    """Metadata of the single materialization emitted by ``node_name``."""
    events = result.asset_materializations_for_node(node_name)
    assert len(events) == 1, f"expected one materialization from {node_name}"
    return events[0].metadata


def read_store(store_uri: str) -> xr.Dataset:
    """Open the published dataset back out of the icechunk store."""
    repo = open_store(store_uri)
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False)


class TestLoadAvailableRetrievals:
    """The tolerant read that stands in for asset inputs."""

    def test_loads_what_is_on_disk(self, tmp_path):
        out = tmp_path / "output"
        for sat in SATELLITES:
            seed_scene(out, sat, T0)
        per_sat = load_available_retrievals(out, SATELLITES, T0)
        assert sorted(per_sat) == sorted(SATELLITES)

    def test_absent_satellite_is_skipped_not_raised(self, tmp_path):
        out = tmp_path / "output"
        seed_scene(out, "goes18", T0)
        per_sat = load_available_retrievals(out, SATELLITES, T0)
        assert list(per_sat) == ["goes18"]

    def test_unreadable_file_is_treated_as_missing(self, tmp_path):
        out = tmp_path / "output"
        seed_scene(out, "goes18", T0)
        truncated = per_satellite_path(out, "goes19", T0)
        truncated.parent.mkdir(parents=True, exist_ok=True)
        truncated.write_bytes(b"not a netcdf file")
        per_sat = load_available_retrievals(out, SATELLITES, T0)
        assert list(per_sat) == ["goes18"]


class TestGlobalMosaicAsset:
    """The mosaic asset, including its tolerance of absent satellites."""

    def test_builds_from_two_satellites(self, tmp_path, tmp_store_uri):
        out = tmp_path / "output"
        seed_scene(out, "goes18", T0, zenith=10.0)
        seed_scene(out, "goes19", T0, zenith=20.0)

        result = materialize(
            [global_mosaic],
            partition_key=key_for(T0),
            resources=build_resources(out, tmp_store_uri),
        )
        assert result.success

        meta = materialization_metadata(result, "global_mosaic")
        assert meta["contributing_satellites"].value == "goes18,goes19"
        assert meta["missing_satellites"].value == "(none)"
        assert meta["n_missing"].value == 0
        assert meta["quality_degraded"].value is False
        assert meta["valid_cells"].value > 0

        mosaic_path = Path(meta["output_path"].value)
        assert mosaic_path.exists()
        with xr.open_dataset(mosaic_path) as ds:
            assert ds["u_wind"].dims == ("latitude", "longitude")
            assert set(ds.attrs) >= {"time", "satellites", "resolution_m",
                                     "quality_degraded", "quality_note"}
            # NetCDF keeps the upstream list; only the icechunk path
            # normalises this attribute to a comma-separated string.
            sats = ds.attrs["satellites"]
            sats = sats.split(",") if isinstance(sats, str) else list(sats)
            assert sats == ["goes18", "goes19"]
            assert ds.attrs["resolution_m"] == pytest.approx(TEST_RESOLUTION_M)
            # Both satellites won cells, and nowhere claims a third one.
            codes = set(ds["source_satellite_index"].values.ravel().tolist())
            assert codes == {-1, 0, 1}

    def test_succeeds_with_one_satellite_missing(self, tmp_path, tmp_store_uri):
        out = tmp_path / "output"
        seed_scene(out, "goes18", T0)

        result = materialize(
            [global_mosaic],
            partition_key=key_for(T0),
            resources=build_resources(out, tmp_store_uri),
        )
        assert result.success

        meta = materialization_metadata(result, "global_mosaic")
        assert meta["contributing_satellites"].value == "goes18"
        assert meta["missing_satellites"].value == "goes19"
        assert meta["n_missing"].value == 1
        assert meta["n_contributing"].value == 1
        assert meta["quality_degraded"].value is True
        assert "goes19" in meta["quality_note"].value
        assert meta["valid_cells"].value > 0

        # The shortfall travels with the data, not only with the run log.
        with xr.open_dataset(Path(meta["output_path"].value)) as ds:
            assert ds.attrs["missing_satellites"] == "goes19"
            assert int(ds.attrs["quality_degraded"]) == 1

    def test_fails_when_no_satellite_produced_anything(self, tmp_path, tmp_store_uri):
        out = tmp_path / "output"
        out.mkdir(parents=True)
        result = materialize(
            [global_mosaic],
            partition_key=key_for(T0),
            resources=build_resources(out, tmp_store_uri),
            raise_on_error=False,
        )
        assert not result.success

    def test_satellite_with_no_usable_pixels_counts_as_missing(
        self, tmp_path, tmp_store_uri,
    ):
        """A file full of rejected pixels is a coverage gap, not a success."""
        out = tmp_path / "output"
        seed_scene(out, "goes18", T0)
        seed_scene(out, "goes19", T0, usable=False)

        result = materialize(
            [global_mosaic],
            partition_key=key_for(T0),
            resources=build_resources(out, tmp_store_uri),
        )
        assert result.success

        meta = materialization_metadata(result, "global_mosaic")
        assert meta["contributing_satellites"].value == "goes18"
        assert meta["missing_satellites"].value == "goes19"
        assert meta["empty_satellites"].value == "goes19"
        assert meta["quality_degraded"].value is True

    def test_fails_when_every_retrieval_is_empty(self, tmp_path, tmp_store_uri):
        """An all-NaN mosaic must not publish as an ordinary timestep."""
        out = tmp_path / "output"
        for sat in SATELLITES:
            seed_scene(out, sat, T0, usable=False)

        result = materialize(
            [global_mosaic],
            partition_key=key_for(T0),
            resources=build_resources(out, tmp_store_uri),
            raise_on_error=False,
        )
        assert not result.success
        # Nothing half-finished was left on disk for a consumer to pick up.
        assert not list(out.rglob("student_amv_global_*.nc"))


class TestAssetFactory:
    """Dependencies and the run-time satellite list come from one source."""

    def test_deps_follow_the_satellites_passed_in(self):
        mosaic_asset, _ = build_mosaic_assets(satellites=("goes18", "mtg-i1"))
        dep_names = {key.to_user_string() for key in mosaic_asset.dependency_keys}
        assert dep_names == {"amv_goes18", "amv_mtg_i1"}

    def test_hyphenated_ids_slugify_to_valid_names(self):
        # Dagster op names must match [A-Za-z0-9_]+.
        assert amv_asset_name("mtg-i1") == "amv_mtg_i1"
        assert amv_asset_name("msg-iodc") == "amv_msg_iodc"

    def test_module_level_assets_use_the_config_defaults(self):
        from operational.config import OperationalConfig

        expected = {amv_asset_name(sat) for sat in OperationalConfig().satellites}
        dep_names = {key.to_user_string() for key in global_mosaic.dependency_keys}
        assert dep_names == expected


class TestPublishedMosaicAsset:
    """Appending mosaics to a real local icechunk store."""

    def test_writes_one_commit_readable_with_time_size_one(
        self, tmp_path, tmp_store_uri,
    ):
        out = tmp_path / "output"
        for sat in SATELLITES:
            seed_scene(out, sat, T0)

        result = materialize(
            [global_mosaic, published_mosaic],
            partition_key=key_for(T0),
            resources=build_resources(out, tmp_store_uri),
        )
        assert result.success

        meta = materialization_metadata(result, "published_mosaic")
        assert meta["written"].value is True
        assert meta["store_uri"].value == tmp_store_uri
        assert meta["time_size"].value == 1
        assert meta["satellite_vocabulary"].value == "goes18,goes19"

        ds = read_store(tmp_store_uri)
        assert ds.sizes["time"] == 1
        assert set(ds.data_vars) >= {"u_wind", "v_wind", "cloud_top_height",
                                     "quality_flag", "sigma_u", "sigma_v",
                                     "sigma_h", "source_satellite_index"}

        # The init snapshot plus exactly one mosaic commit.
        repo = open_store(tmp_store_uri)
        assert len(list(repo.ancestry(branch="main"))) == 2

    def test_second_partition_appends_in_order(self, tmp_path, tmp_store_uri):
        out = tmp_path / "output"
        resources = build_resources(out, tmp_store_uri)
        for t in (T0, T1):
            for sat in SATELLITES:
                seed_scene(out, sat, t)
            result = materialize(
                [global_mosaic, published_mosaic],
                partition_key=key_for(t),
                resources=resources,
            )
            assert result.success

        ds = read_store(tmp_store_uri)
        assert ds.sizes["time"] == 2
        times = list(ds["time"].values)
        assert len(set(times)) == 2
        assert times == sorted(times)

        repo = open_store(tmp_store_uri)
        assert len(list(repo.ancestry(branch="main"))) == 3

    def test_republishing_a_partition_skips(self, tmp_path, tmp_store_uri):
        out = tmp_path / "output"
        resources = build_resources(out, tmp_store_uri)
        for sat in SATELLITES:
            seed_scene(out, sat, T0)

        first = materialize(
            [global_mosaic, published_mosaic],
            partition_key=key_for(T0),
            resources=resources,
        )
        assert materialization_metadata(first, "published_mosaic")["written"].value

        second = materialize(
            [global_mosaic, published_mosaic],
            partition_key=key_for(T0),
            resources=resources,
        )
        assert second.success
        meta = materialization_metadata(second, "published_mosaic")
        assert meta["written"].value is False
        assert "already in the store" in meta["skipped_reason"].value
        assert meta["time_size"].value == 1

        ds = read_store(tmp_store_uri)
        assert ds.sizes["time"] == 1

        # No second commit: the store is byte-for-byte where it was.
        repo = open_store(tmp_store_uri)
        assert len(list(repo.ancestry(branch="main"))) == 2

    def test_retry_policy_is_set(self):
        """Object-store writes are network operations, so they get retries."""
        policy = published_mosaic.op.retry_policy
        assert policy is not None
        assert policy.max_retries >= 1


class TestEndToEnd:
    """Mosaic and publish, together, over a real store."""

    def test_materialize_mosaic_and_publish(self, tmp_path, tmp_store_uri):
        out = tmp_path / "output"
        seed_scene(out, "goes18", T0, zenith=5.0)
        seed_scene(out, "goes19", T0, zenith=30.0)

        result = materialize(
            [global_mosaic, published_mosaic],
            partition_key=key_for(T0),
            resources=build_resources(out, tmp_store_uri),
        )
        assert result.success

        ds = read_store(tmp_store_uri)
        assert ds.sizes["time"] == 1
        assert ds["time"].values[0].astype("datetime64[s]").item() == T0
        flag_meanings = ds["source_satellite_index"].attrs["flag_meanings"].split()
        assert flag_meanings == ["goes18", "goes19"]
        assert int(ds["u_wind"].isel(time=0).notnull().sum().compute()) > 0

    def test_partial_cycle_still_publishes(self, tmp_path, tmp_store_uri):
        """One satellite down must not cost the cycle its published timestep."""
        out = tmp_path / "output"
        seed_scene(out, "goes19", T0)

        result = materialize(
            [global_mosaic, published_mosaic],
            partition_key=key_for(T0),
            resources=build_resources(out, tmp_store_uri),
        )
        assert result.success

        ds = read_store(tmp_store_uri)
        assert ds.sizes["time"] == 1
        assert ds.attrs["satellites"] == "goes19"
        assert int(ds.attrs["quality_degraded"]) == 1
