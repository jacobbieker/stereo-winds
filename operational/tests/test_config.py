"""Tests for :mod:`operational.config` and the shared synthetic scene."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from operational.config import ENV_PREFIX, OperationalConfig
from operational.tests.conftest import AMV_VARS, synthetic_scene


class TestDefaults:
    """The out-of-the-box configuration."""

    def test_default_satellites(self):
        """The full ring, including the two icechunk-only satellites."""
        cfg = OperationalConfig()
        assert cfg.satellites == ("goes18", "goes19", "himawari9", "gk2a", "mtg-i1", "msg-iodc")

    def test_required_satellites_excludes_the_sparse_ones(self):
        """MTG and IODC get assets, but a run does not wait for them."""
        cfg = OperationalConfig()
        assert cfg.required_satellites == ("goes18", "goes19", "himawari9", "gk2a")
        assert set(cfg.required_satellites) <= set(cfg.satellites)

    def test_narrowing_the_ring_narrows_what_is_required(self):
        cfg = OperationalConfig().with_satellites("goes19", "mtg-i1")
        assert cfg.satellites == ("goes19", "mtg-i1")
        assert cfg.required_satellites == ("goes19",)

    def test_a_ring_of_only_sparse_satellites_requires_them(self):
        """Otherwise nothing would ever be required, and nothing would run."""
        cfg = OperationalConfig(satellites=("mtg-i1", "msg-iodc"))
        assert cfg.required_satellites == ("mtg-i1", "msg-iodc")

    def test_required_must_name_satellites_in_the_ring(self):
        with pytest.raises(ValueError, match="not in the ring"):
            OperationalConfig(satellites=("goes19",), required_satellites=("gk2a",))

    def test_default_bands_match_student_dataset(self):
        from stereo_winds.student_dataset import (
            DEFAULT_FLOW_BANDS,
            DEFAULT_RAD_BANDS,
        )

        cfg = OperationalConfig()
        assert cfg.flow_bands == tuple(DEFAULT_FLOW_BANDS)
        assert cfg.rad_bands == tuple(DEFAULT_RAD_BANDS)

    def test_default_scalars(self):
        cfg = OperationalConfig()
        assert cfg.cadence_minutes == 60
        assert cfg.availability_tolerance_minutes == 5.0
        assert cfg.output_dir == Path("output/operational")
        assert cfg.store_uri == "output/operational.icechunk"
        assert cfg.resolution_m == 10000.0
        assert cfg.device == "cpu"
        assert cfg.row_strip == 1024

    def test_is_frozen(self):
        cfg = OperationalConfig()
        with pytest.raises(Exception):
            cfg.device = "cuda"  # type: ignore[misc]

    def test_derived_timedeltas(self):
        cfg = OperationalConfig(cadence_minutes=10, availability_tolerance_minutes=2.5)
        assert cfg.cadence == timedelta(minutes=10)
        assert cfg.availability_tolerance == timedelta(minutes=2.5)


class TestOverrides:
    """Explicit construction, coercion and validation."""

    def test_sequences_are_coerced_to_tuples(self):
        cfg = OperationalConfig(satellites=["goes19"], flow_bands=["C14"], rad_bands=["C13", "C14"])
        assert cfg.satellites == ("goes19",)
        assert cfg.flow_bands == ("C14",)
        assert cfg.rad_bands == ("C13", "C14")

    def test_output_dir_string_becomes_path(self):
        cfg = OperationalConfig(output_dir="/data/amv")
        assert cfg.output_dir == Path("/data/amv")
        assert isinstance(cfg.output_dir, Path)

    def test_numeric_fields_are_cast(self):
        cfg = OperationalConfig(
            cadence_minutes="15",
            resolution_m="2000",
            row_strip="64",
            availability_tolerance_minutes="1",
        )
        assert cfg.cadence_minutes == 15
        assert cfg.resolution_m == 2000.0
        assert cfg.row_strip == 64
        assert cfg.availability_tolerance_minutes == 1.0

    def test_with_satellites_returns_copy(self):
        cfg = OperationalConfig()
        narrowed = cfg.with_satellites("goes19")
        assert narrowed.satellites == ("goes19",)
        assert cfg.satellites == ("goes18", "goes19", "himawari9", "gk2a", "mtg-i1", "msg-iodc")
        assert narrowed.store_uri == cfg.store_uri

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"satellites": ()},
            {"satellites": ("goes19", "goes19")},
            {"cadence_minutes": 0},
            {"cadence_minutes": -10},
            {"availability_tolerance_minutes": -1.0},
            {"resolution_m": 0.0},
            {"row_strip": 0},
            {"flow_bands": ()},
            {"rad_bands": ()},
            {"store_uri": ""},
        ],
    )
    def test_invalid_values_rejected(self, kwargs):
        with pytest.raises(ValueError):
            OperationalConfig(**kwargs)


class TestStoreUri:
    """Local path versus object-storage destinations."""

    def test_local_path(self):
        cfg = OperationalConfig(store_uri="output/amv.icechunk")
        assert cfg.store_is_s3 is False
        assert cfg.store_path == Path("output/amv.icechunk")

    def test_s3_uri(self):
        cfg = OperationalConfig(store_uri="s3://my-bucket/amv/global")
        assert cfg.store_is_s3 is True
        assert cfg.store_path is None
        assert cfg.store_bucket_prefix() == ("my-bucket", "amv/global")

    def test_s3_uri_without_prefix(self):
        cfg = OperationalConfig(store_uri="s3://my-bucket")
        assert cfg.store_bucket_prefix() == ("my-bucket", "")

    def test_s3_trailing_slash_trimmed(self):
        cfg = OperationalConfig(store_uri="s3://my-bucket/amv/")
        assert cfg.store_bucket_prefix() == ("my-bucket", "amv")

    def test_bucket_prefix_rejects_local_path(self):
        cfg = OperationalConfig(store_uri="output/amv.icechunk")
        with pytest.raises(ValueError, match="local path"):
            cfg.store_bucket_prefix()

    def test_bucket_prefix_rejects_empty_bucket(self):
        cfg = OperationalConfig(store_uri="s3:///amv")
        with pytest.raises(ValueError, match="no bucket"):
            cfg.store_bucket_prefix()

    def test_scheme_is_case_insensitive(self):
        cfg = OperationalConfig(store_uri="S3://my-bucket/amv")
        assert cfg.store_is_s3 is True
        assert cfg.store_path is None
        assert cfg.store_bucket_prefix() == ("my-bucket", "amv")

    @pytest.mark.parametrize("uri", ["gs://bucket/amv", "az://bucket/amv", "http://host/amv"])
    def test_unsupported_scheme_rejected(self, uri):
        with pytest.raises(ValueError, match="unsupported scheme"):
            OperationalConfig(store_uri=uri)

    def test_absolute_path_is_not_a_scheme(self):
        cfg = OperationalConfig(store_uri="/data/amv/global.icechunk")
        assert cfg.store_is_s3 is False
        assert cfg.store_path == Path("/data/amv/global.icechunk")


class TestSatelliteValidation:
    """Unknown ids warn rather than blowing up construction."""

    def test_known_ids_do_not_warn(self, caplog):
        with caplog.at_level("WARNING", logger="operational.config"):
            OperationalConfig(satellites=("goes19", "gk2a"))
        assert caplog.records == []

    def test_unknown_id_warns_but_builds(self, caplog):
        with caplog.at_level("WARNING", logger="operational.config"):
            cfg = OperationalConfig(satellites=("goes19", "goes-19"))
        assert cfg.satellites == ("goes19", "goes-19")
        assert any("goes-19" in r.getMessage() for r in caplog.records)


class TestTimestamps:
    """Cadence alignment and backfill windows."""

    def test_floor_hourly(self):
        cfg = OperationalConfig(cadence_minutes=60)
        assert cfg.floor_to_cadence(datetime(2024, 1, 15, 12, 47, 13)) == datetime(
            2024, 1, 15, 12, 0
        )

    def test_floor_ten_minutes(self):
        cfg = OperationalConfig(cadence_minutes=10)
        assert cfg.floor_to_cadence(datetime(2024, 1, 15, 12, 47, 13)) == datetime(
            2024, 1, 15, 12, 40
        )

    def test_floor_is_idempotent(self):
        cfg = OperationalConfig(cadence_minutes=30)
        aligned = cfg.floor_to_cadence(datetime(2024, 1, 15, 12, 47))
        assert cfg.floor_to_cadence(aligned) == aligned

    def test_timestamps_inclusive_range(self):
        cfg = OperationalConfig(cadence_minutes=60)
        stamps = list(cfg.timestamps(datetime(2024, 1, 15, 0), datetime(2024, 1, 15, 3)))
        assert stamps == [datetime(2024, 1, 15, h) for h in range(4)]

    def test_timestamps_skip_partial_leading_slot(self):
        cfg = OperationalConfig(cadence_minutes=60)
        stamps = list(cfg.timestamps(datetime(2024, 1, 15, 0, 30), datetime(2024, 1, 15, 2, 15)))
        assert stamps == [datetime(2024, 1, 15, 1), datetime(2024, 1, 15, 2)]

    def test_grid_is_continuous_across_midnight(self):
        """A cadence that does not divide a day must not re-anchor at 00:00."""
        cfg = OperationalConfig(cadence_minutes=50)
        stamps = list(cfg.timestamps(datetime(2024, 1, 15, 22, 0), datetime(2024, 1, 16, 1, 0)))
        # Every slot the enumerator yields must also be a fixed point of
        # the flooring the sensor uses, on both sides of midnight.
        assert stamps
        for stamp in stamps:
            assert cfg.floor_to_cadence(stamp) == stamp
        gaps = {b - a for a, b in zip(stamps, stamps[1:])}
        assert gaps == {timedelta(minutes=50)}

    def test_tz_aware_floor_uses_absolute_time(self):
        from datetime import timezone

        cfg = OperationalConfig(cadence_minutes=60)
        when = datetime(2024, 1, 15, 12, 47, tzinfo=timezone.utc)
        floored = cfg.floor_to_cadence(when)
        assert floored == datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        assert floored.tzinfo is timezone.utc

    def test_timestamps_empty_when_reversed(self):
        cfg = OperationalConfig()
        assert list(cfg.timestamps(datetime(2024, 1, 16), datetime(2024, 1, 15))) == []


class TestPaths:
    """Derived output paths."""

    def test_timestamp_key(self):
        assert OperationalConfig.timestamp_key(datetime(2024, 1, 15, 12, 0)) == "20240115T120000"

    def test_satellite_and_mosaic_paths(self, tmp_path):
        cfg = OperationalConfig(output_dir=tmp_path / "op")
        t0 = datetime(2024, 1, 15, 12, 0)
        sat = cfg.satellite_path("goes19", t0)
        mosaic = cfg.mosaic_path(t0)
        assert sat == tmp_path / "op" / "20240115T120000" / "amv_goes19_20240115T120000.nc"
        assert mosaic == tmp_path / "op" / "20240115T120000" / "mosaic_20240115T120000.nc"
        assert sat.parent == mosaic.parent

    def test_satellite_paths_differ_per_satellite(self):
        cfg = OperationalConfig()
        t0 = datetime(2024, 1, 15, 12, 0)
        assert cfg.satellite_path("goes19", t0) != cfg.satellite_path("gk2a", t0)


class TestFromEnv:
    """Environment-driven construction."""

    def test_defaults_when_env_empty(self):
        assert OperationalConfig.from_env(env={}) == OperationalConfig()

    def test_reads_all_fields(self):
        env = {
            ENV_PREFIX + "SATELLITES": "goes19, himawari9",
            ENV_PREFIX + "FLOW_BANDS": "C08,C14",
            ENV_PREFIX + "RAD_BANDS": "C13,C14",
            ENV_PREFIX + "CADENCE_MINUTES": "10",
            ENV_PREFIX + "AVAILABILITY_TOLERANCE_MINUTES": "2.5",
            ENV_PREFIX + "OUTPUT_DIR": "/data/op",
            ENV_PREFIX + "STORE_URI": "s3://bucket/amv",
            ENV_PREFIX + "RESOLUTION_M": "2000",
            ENV_PREFIX + "DEVICE": "cuda",
            ENV_PREFIX + "ROW_STRIP": "512",
        }
        cfg = OperationalConfig.from_env(env=env)
        assert cfg.satellites == ("goes19", "himawari9")
        assert cfg.flow_bands == ("C08", "C14")
        assert cfg.rad_bands == ("C13", "C14")
        assert cfg.cadence_minutes == 10
        assert cfg.availability_tolerance_minutes == 2.5
        assert cfg.output_dir == Path("/data/op")
        assert cfg.store_uri == "s3://bucket/amv"
        assert cfg.resolution_m == 2000.0
        assert cfg.device == "cuda"
        assert cfg.row_strip == 512

    def test_blank_value_falls_back_to_default(self):
        cfg = OperationalConfig.from_env(env={ENV_PREFIX + "DEVICE": "  "})
        assert cfg.device == "cpu"

    def test_overrides_beat_environment(self):
        cfg = OperationalConfig.from_env(env={ENV_PREFIX + "DEVICE": "cuda"}, device="cpu")
        assert cfg.device == "cpu"

    def test_reads_os_environ_by_default(self, monkeypatch):
        monkeypatch.setenv(ENV_PREFIX + "CADENCE_MINUTES", "15")
        assert OperationalConfig.from_env().cadence_minutes == 15


class TestOpConfigFixture:
    """The shared fixtures keep everything under ``tmp_path``."""

    def test_paths_are_under_tmp_path(self, op_config, tmp_path):
        assert tmp_path in op_config.output_dir.parents or op_config.output_dir.parent == tmp_path
        assert str(tmp_path) in op_config.store_uri
        assert op_config.store_is_s3 is False

    def test_tmp_store_uri(self, tmp_store_uri, tmp_path):
        assert tmp_store_uri.startswith(str(tmp_path))
        assert not Path(tmp_store_uri).exists()


class TestSyntheticScene:
    """The synthetic AMV scene matches the real per-satellite schema."""

    def test_schema(self):
        ds = synthetic_scene("goes19", datetime(2024, 1, 15, 12, 0))
        assert list(ds.data_vars) == AMV_VARS
        assert AMV_VARS == [
            "u_wind",
            "v_wind",
            "cloud_top_height",
            "quality_flag",
            "sigma_u",
            "sigma_v",
            "sigma_h",
        ]
        for name in AMV_VARS:
            assert ds[name].dims == ("y", "x")
            assert ds[name].dtype == np.float32
            assert np.isfinite(ds[name].values).all()

    def test_matches_upstream_output_vars(self):
        """The variable list is the one ``infer_satellite`` writes."""
        upstream = [
            "u_wind",
            "v_wind",
            "cloud_top_height",
            "quality_flag",
            "sigma_u",
            "sigma_v",
            "sigma_h",
        ]
        assert AMV_VARS == upstream

    def test_shape(self):
        ds = synthetic_scene("gk2a", datetime(2024, 1, 15, 12, 0), ny=8, nx=12)
        assert ds.sizes == {"y": 8, "x": 12}

    def test_coords(self):
        ds = synthetic_scene("himawari9", datetime(2024, 1, 15, 12, 0), ny=16, nx=16)
        for coord in ("latitude", "longitude", "zenith_angle"):
            assert ds[coord].dims == ("y", "x")
            assert ds[coord].shape == (16, 16)
            assert ds[coord].dtype == np.float32
        assert np.abs(ds["latitude"].values).max() <= 90.0
        lon = ds["longitude"].values
        assert lon.min() >= -180.0 and lon.max() <= 180.0

    def test_quality_flag_is_two(self):
        ds = synthetic_scene("goes18", datetime(2024, 1, 15, 12, 0))
        assert np.all(ds["quality_flag"].values == 2.0)

    def test_attrs(self):
        t0 = datetime(2024, 1, 15, 12, 0)
        ds = synthetic_scene("goes18", t0)
        assert ds.attrs["satellite_id"] == "goes18"
        assert ds.attrs["time"] == str(t0)
        assert ds.attrs["source"] == "student_amv"

    def test_zenith_is_constant_and_configurable(self):
        ds = synthetic_scene("goes19", datetime(2024, 1, 15, 12, 0), zenith=42.5)
        assert np.all(ds["zenith_angle"].values == np.float32(42.5))

    def test_default_zenith(self):
        ds = synthetic_scene("goes19", datetime(2024, 1, 15, 12, 0))
        assert np.all(ds["zenith_angle"].values == np.float32(10.0))

    def test_deterministic(self):
        t0 = datetime(2024, 1, 15, 12, 0)
        a = synthetic_scene("goes19", t0)
        b = synthetic_scene("goes19", t0)
        for name in AMV_VARS:
            np.testing.assert_array_equal(a[name].values, b[name].values)

    def test_satellites_differ(self):
        t0 = datetime(2024, 1, 15, 12, 0)
        a = synthetic_scene("goes19", t0)
        b = synthetic_scene("himawari9", t0)
        assert not np.array_equal(a["u_wind"].values, b["u_wind"].values)
        # Different sub-satellite longitudes place the footprints apart.
        assert not np.allclose(a["longitude"].values, b["longitude"].values)

    def test_unknown_satellite_centred_on_prime_meridian(self):
        ds = synthetic_scene("not-a-satellite", datetime(2024, 1, 15, 12, 0), ny=4, nx=4)
        assert ds.attrs["satellite_id"] == "not-a-satellite"
        assert abs(float(ds["longitude"].values.mean())) < 1e-3

    @pytest.mark.parametrize("ny,nx", [(0, 4), (4, 0), (-1, 4)])
    def test_rejects_empty_grid(self, ny, nx):
        with pytest.raises(ValueError):
            synthetic_scene("goes19", datetime(2024, 1, 15, 12, 0), ny=ny, nx=nx)
