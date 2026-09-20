"""Tests for the operational icechunk publish step.

Every store here is a real local icechunk store under ``tmp_path`` — no
S3, no network.  The S3 path is exercised only as far as constructing the
storage object.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("icechunk")

from stereo_winds.icechunk_output import NO_SOURCE  # noqa: E402

from operational.core.publish import (  # noqa: E402
    PublishResult,
    as_store_time,
    existing_timestamps,
    open_store,
    publish_mosaic,
    seed_vocabulary,
)

T0 = datetime(2026, 7, 28, 0, 0)
VARS = ["u_wind", "v_wind", "cloud_top_height", "quality_flag",
        "sigma_u", "sigma_v", "sigma_h"]


def _mosaic(satellites, codes=None, n=4, value=1.0) -> xr.Dataset:
    """A global mosaic carrying its own satellite vocabulary."""
    ds = xr.Dataset(
        {v: (("latitude", "longitude"), np.full((n, n), value, np.float32))
         for v in VARS},
        coords={"latitude": np.linspace(-80.0, 80.0, n),
                "longitude": np.linspace(-170.0, 170.0, n)},
        attrs={"title": "Global student AMV mosaic",
               "resolution_m": 10000.0,
               "satellites": list(satellites)},
    )
    if codes is None:
        # Cycle the codes across the grid so every satellite contributes.
        codes = np.tile(np.arange(len(satellites), dtype=np.int8),
                        (n, n // len(satellites) + 1))[:, :n]
    ds["source_satellite_index"] = (("latitude", "longitude"),
                                    np.asarray(codes, np.int8))
    ds["source_satellite_index"].attrs.update({
        "flag_values": list(range(len(satellites))),
        "flag_meanings": " ".join(satellites),
        "no_source_index": NO_SOURCE,
    })
    return ds


def _decode(codes: np.ndarray, names: list[str]) -> np.ndarray:
    """Satellite name per cell, so two encodings can be compared."""
    out = np.full(np.shape(codes), "", dtype="U12")
    for code, name in enumerate(names):
        out[np.asarray(codes) == code] = name
    return out


def _stored(repo) -> xr.Dataset:
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False)


def _stored_names(stored: xr.Dataset) -> list[str]:
    return str(stored["source_satellite_index"].attrs["flag_meanings"]).split()


def _times(stored: xr.Dataset) -> list[datetime]:
    return [pd.Timestamp(v).to_pydatetime() for v in stored["time"].values]


@pytest.fixture
def repo(tmp_store_uri):
    return open_store(tmp_store_uri)


class TestOpenStore:
    def test_a_local_path_creates_the_store(self, tmp_path):
        uri = str(tmp_path / "nested" / "store.icechunk")
        repo = open_store(uri)
        assert repo is not None
        assert (tmp_path / "nested" / "store.icechunk").is_dir()

    def test_reopening_a_local_store_sees_its_contents(self, tmp_store_uri):
        publish_mosaic(open_store(tmp_store_uri), _mosaic(["goes19"]), T0)
        assert existing_timestamps(open_store(tmp_store_uri)) == {T0}

    def test_an_s3_uri_builds_s3_storage(self, monkeypatch):
        """No network: only that the S3 storage object is constructed."""
        import icechunk

        captured = {}

        class _Repo:
            @staticmethod
            def open_or_create(storage):
                captured["storage"] = storage
                return "repo"

        monkeypatch.setattr(icechunk, "Repository", _Repo)
        repo = open_store(
            "s3://a-bucket/a/prefix", region="us-east-1", anonymous=True,
            endpoint_url="https://example.invalid", force_path_style=True,
        )
        assert repo == "repo"
        assert isinstance(captured["storage"], icechunk.Storage)

    def test_an_unsupported_scheme_is_rejected(self):
        with pytest.raises(ValueError):
            open_store("gs://a-bucket/a/prefix")


class TestFirstPublish:
    def test_it_reports_what_it_wrote(self, repo):
        result = publish_mosaic(repo, _mosaic(["goes18", "goes19"]), T0)
        assert isinstance(result, PublishResult)
        assert (result.written, result.skipped_reason) == (True, None)
        assert result.timestamp == T0
        assert result.vocabulary == ("goes18", "goes19")

    def test_the_mosaic_reads_back_identically(self, repo):
        source = _mosaic(["goes18"], value=3.5)
        publish_mosaic(repo, source, T0)

        stored = _stored(repo).isel(time=0)
        for var in VARS:
            assert np.array_equal(stored[var].values, source[var].values,
                                  equal_nan=True), var
        assert np.array_equal(stored["source_satellite_index"].values,
                              source["source_satellite_index"].values)
        assert np.allclose(stored["latitude"].values, source["latitude"].values)
        assert np.allclose(stored["longitude"].values,
                           source["longitude"].values)
        assert _times(_stored(repo)) == [T0]

    def test_the_store_starts_empty(self, repo):
        assert existing_timestamps(repo) == set()


class TestAppend:
    def test_a_later_timestamp_grows_the_time_dimension(self, repo):
        t1 = T0 + timedelta(hours=6)
        vocabulary: list[str] = []
        publish_mosaic(repo, _mosaic(["goes19"]), T0, vocabulary=vocabulary)
        result = publish_mosaic(repo, _mosaic(["goes19"]), t1,
                                vocabulary=vocabulary)

        assert result.written is True
        stored = _stored(repo)
        assert stored.sizes["time"] == 2
        assert _times(stored) == [T0, t1]

    def test_appended_values_stay_with_their_timestep(self, repo):
        t1 = T0 + timedelta(hours=6)
        publish_mosaic(repo, _mosaic(["goes19"], value=1.0), T0)
        publish_mosaic(repo, _mosaic(["goes19"], value=2.0), t1)

        stored = _stored(repo)
        assert np.allclose(stored["u_wind"].isel(time=0).values, 1.0)
        assert np.allclose(stored["u_wind"].isel(time=1).values, 2.0)

    def test_the_time_coordinate_survives_many_appends(self, repo):
        """Regression: appends re-encoding against the first write's units."""
        times = [T0, T0 + timedelta(minutes=10), T0 + timedelta(hours=6),
                 T0 + timedelta(days=3, seconds=30)]
        vocabulary: list[str] = []
        for when in times:
            publish_mosaic(repo, _mosaic(["goes19"]), when,
                           vocabulary=vocabulary)

        stored = _stored(repo)
        assert _times(stored) == times
        assert len(set(_times(stored))) == len(times)
        assert existing_timestamps(repo) == set(times)


class TestIdempotence:
    def test_republishing_a_timestamp_is_skipped(self, repo):
        publish_mosaic(repo, _mosaic(["goes19"]), T0)
        result = publish_mosaic(repo, _mosaic(["goes19"]), T0)

        assert result.written is False
        assert result.skipped_reason and "already" in result.skipped_reason
        assert _stored(repo).sizes["time"] == 1
        assert _times(_stored(repo)) == [T0]

    def test_a_rerun_over_a_mixed_batch_writes_only_what_is_new(self, repo):
        times = [T0 + timedelta(hours=6 * i) for i in range(3)]
        vocabulary: list[str] = []
        for when in times[:2]:
            publish_mosaic(repo, _mosaic(["goes19"]), when,
                           vocabulary=vocabulary)

        results = [publish_mosaic(repo, _mosaic(["goes19"]), when,
                                  vocabulary=vocabulary) for when in times]
        assert [r.written for r in results] == [False, False, True]
        assert _times(_stored(repo)) == times

    def test_a_skipped_publish_still_reports_the_vocabulary(self, repo):
        publish_mosaic(repo, _mosaic(["goes18", "goes19"]), T0)
        result = publish_mosaic(repo, _mosaic(["goes18", "goes19"]), T0)
        assert result.vocabulary == ("goes18", "goes19")

    def test_skip_existing_false_replaces_rather_than_duplicating(self, repo):
        """Writing anyway overwrites the timestep; it never doubles it."""
        publish_mosaic(repo, _mosaic(["goes19"], value=1.0), T0)
        result = publish_mosaic(repo, _mosaic(["goes19"], value=2.0), T0,
                                skip_existing=False)
        assert result.written is True
        assert result.action == "replaced"
        stored = _stored(repo)
        assert stored.sizes["time"] == 1
        assert _times(stored) == [T0]
        assert float(stored["u_wind"].values[0].flat[0]) == 2.0

    def test_an_aware_timestamp_matches_what_the_store_holds(self, repo):
        """Regression: an orchestrator's tz-aware UTC never compared equal."""
        publish_mosaic(repo, _mosaic(["goes19"]), T0)
        result = publish_mosaic(repo, _mosaic(["goes19"]),
                                T0.replace(tzinfo=timezone.utc))
        assert result.written is False
        assert _stored(repo).sizes["time"] == 1

    def test_an_aware_timestamp_is_stored_as_utc(self, repo):
        """A non-UTC offset must not be written at its wall-clock value."""
        eastern = timezone(timedelta(hours=-5))
        aware = (T0 + timedelta(hours=6)).replace(tzinfo=eastern)
        result = publish_mosaic(repo, _mosaic(["goes19"]), aware)
        assert result.timestamp == T0 + timedelta(hours=11)
        assert _times(_stored(repo)) == [T0 + timedelta(hours=11)]

    def test_as_store_time_leaves_a_naive_timestamp_alone(self):
        assert as_store_time(T0) == T0
        assert as_store_time(T0.replace(tzinfo=timezone.utc)) == T0


class TestOrdering:
    def test_an_older_timestamp_is_refused(self, repo):
        """An unsorted time index makes range selection return nothing."""
        publish_mosaic(repo, _mosaic(["goes19"]), T0)
        with pytest.raises(ValueError, match="chronological"):
            publish_mosaic(repo, _mosaic(["goes19"]), T0 - timedelta(hours=6))
        assert _times(_stored(repo)) == [T0]

    def test_out_of_order_can_be_opted_into(self, repo):
        publish_mosaic(repo, _mosaic(["goes19"]), T0)
        result = publish_mosaic(repo, _mosaic(["goes19"]),
                                T0 - timedelta(hours=6),
                                allow_out_of_order=True)
        assert result.written is True
        assert _stored(repo).sizes["time"] == 2

    def test_publishing_in_order_stays_selectable_by_range(self, repo):
        times = [T0 + timedelta(hours=6 * i) for i in range(3)]
        vocabulary: list[str] = []
        for when in times:
            publish_mosaic(repo, _mosaic(["goes19"]), when,
                           vocabulary=vocabulary)
        stored = _stored(repo)
        assert stored.indexes["time"].is_monotonic_increasing
        assert stored.sel(time=slice(times[0], times[1])).sizes["time"] == 2


class TestSharedVocabulary:
    def test_provenance_survives_differing_satellite_lists(self, repo):
        """Code 2 must not mean gk2a in one timestep and himawari9 in the next."""
        specs = [(["goes18", "goes19"], T0),
                 (["goes19", "himawari9"], T0 + timedelta(hours=6))]
        vocabulary: list[str] = []
        sources = []
        for names, when in specs:
            ds = _mosaic(names)
            sources.append((names, ds["source_satellite_index"].values.copy()))
            publish_mosaic(repo, ds, when, vocabulary=vocabulary)

        stored = _stored(repo)
        names_in_store = _stored_names(stored)
        assert names_in_store == ["goes18", "goes19", "himawari9"]
        for step, (names, codes) in enumerate(sources):
            got = stored["source_satellite_index"].isel(time=step).values
            assert np.array_equal(_decode(codes, names),
                                  _decode(got, names_in_store)), step

    def test_the_vocabulary_is_extended_in_place(self, repo):
        vocabulary: list[str] = []
        publish_mosaic(repo, _mosaic(["goes18", "goes19"]), T0,
                       vocabulary=vocabulary)
        assert vocabulary == ["goes18", "goes19"]
        publish_mosaic(repo, _mosaic(["gk2a", "goes18"]),
                       T0 + timedelta(hours=6), vocabulary=vocabulary)
        assert vocabulary == ["goes18", "goes19", "gk2a"]

    def test_a_resumed_run_continues_the_stores_vocabulary(self, repo):
        """A fresh process starts with an empty list; earlier codes must hold."""
        first = _mosaic(["goes18", "goes19"])
        publish_mosaic(repo, first, T0, vocabulary=[])

        # Second "process": nothing carried over but the store itself.
        later = _mosaic(["himawari9", "goes18"])
        result = publish_mosaic(repo, later, T0 + timedelta(hours=6),
                                vocabulary=[])
        assert result.vocabulary == ("goes18", "goes19", "himawari9")

        stored = _stored(repo)
        names_in_store = _stored_names(stored)
        assert np.array_equal(
            _decode(first["source_satellite_index"].values,
                    ["goes18", "goes19"]),
            _decode(stored["source_satellite_index"].isel(time=0).values,
                    names_in_store))
        assert np.array_equal(
            _decode(later["source_satellite_index"].values,
                    ["himawari9", "goes18"]),
            _decode(stored["source_satellite_index"].isel(time=1).values,
                    names_in_store))

    def test_the_no_source_sentinel_survives(self, repo):
        codes = np.array([[NO_SOURCE, 0, 1, NO_SOURCE]] * 4, np.int8)
        publish_mosaic(repo, _mosaic(["goes18", "goes19"], codes=codes), T0)
        stored = _stored(repo)["source_satellite_index"].isel(time=0).values
        assert stored.dtype == np.int8
        assert np.array_equal(stored, codes)

    def test_seeding_keeps_the_callers_list_object(self, repo):
        publish_mosaic(repo, _mosaic(["goes18"]), T0)
        vocabulary = ["himawari9"]
        seeded = seed_vocabulary(repo, vocabulary)
        assert seeded is vocabulary
        assert vocabulary == ["goes18", "himawari9"]

    def test_seeding_a_compatible_list_leaves_it_alone(self, repo):
        publish_mosaic(repo, _mosaic(["goes18"]), T0)
        vocabulary = ["goes18", "gk2a"]
        assert seed_vocabulary(repo, vocabulary) == ["goes18", "gk2a"]

    def test_seeding_none_reads_the_store(self, repo):
        publish_mosaic(repo, _mosaic(["goes18", "goes19"]), T0)
        assert seed_vocabulary(repo, None) == ["goes18", "goes19"]


class TestRepairingADegradedTimestep:
    """A resumed satellite must be able to correct what was published."""

    def test_a_mosaic_that_adds_a_satellite_replaces_the_stored_one(self, repo):
        """The whole point of resuming one satellite at a time."""
        vocabulary: list[str] = []
        degraded = publish_mosaic(repo, _mosaic(["goes18", "goes19"]), T0,
                                  vocabulary=vocabulary)
        assert degraded.action == "created"

        repaired = publish_mosaic(
            repo, _mosaic(["goes18", "goes19", "gk2a"]), T0,
            vocabulary=vocabulary)
        assert repaired.written is True
        assert repaired.action == "replaced"

        stored = _stored(repo)
        assert stored.sizes["time"] == 1, "a repair must not add a timestep"
        assert set(str(stored["satellites_contributing"].values[0]).split(",")) \
            == {"goes18", "goes19", "gk2a"}

    def test_a_mosaic_that_adds_nothing_is_still_skipped(self, repo):
        """Ordinary re-runs must stay no-ops."""
        vocabulary: list[str] = []
        publish_mosaic(repo, _mosaic(["goes18", "goes19"]), T0,
                       vocabulary=vocabulary)
        again = publish_mosaic(repo, _mosaic(["goes18", "goes19"]), T0,
                               vocabulary=vocabulary)
        assert again.written is False
        assert again.action == "skipped"
        assert _stored(repo).sizes["time"] == 1

    def test_a_mosaic_with_fewer_satellites_does_not_overwrite_a_better_one(
            self, repo):
        """A later outage must not undo a complete timestep."""
        vocabulary: list[str] = []
        publish_mosaic(repo, _mosaic(["goes18", "goes19", "gk2a"]), T0,
                       vocabulary=vocabulary)
        worse = publish_mosaic(repo, _mosaic(["goes18"]), T0,
                               vocabulary=vocabulary)
        assert worse.written is False
        stored = _stored(repo)
        assert set(str(stored["satellites_contributing"].values[0]).split(",")) \
            == {"goes18", "goes19", "gk2a"}

    def test_repair_can_be_turned_off(self, repo):
        vocabulary: list[str] = []
        publish_mosaic(repo, _mosaic(["goes18"]), T0, vocabulary=vocabulary)
        result = publish_mosaic(repo, _mosaic(["goes18", "gk2a"]), T0,
                                vocabulary=vocabulary, repair_improved=False)
        assert result.written is False
        assert result.action == "skipped"

    def test_replace_existing_forces_a_rewrite_that_adds_nothing(self, repo):
        vocabulary: list[str] = []
        publish_mosaic(repo, _mosaic(["goes18"], value=1.0), T0,
                       vocabulary=vocabulary)
        result = publish_mosaic(repo, _mosaic(["goes18"], value=5.0), T0,
                                vocabulary=vocabulary, replace_existing=True)
        assert result.action == "replaced"
        stored = _stored(repo)
        assert stored.sizes["time"] == 1
        assert float(stored["u_wind"].values[0].flat[0]) == 5.0

    def test_a_repair_leaves_later_timesteps_alone(self, repo):
        vocabulary: list[str] = []
        publish_mosaic(repo, _mosaic(["goes18"], value=1.0), T0,
                       vocabulary=vocabulary)
        later = T0 + timedelta(hours=6)
        publish_mosaic(repo, _mosaic(["goes18", "goes19"], value=2.0), later,
                       vocabulary=vocabulary)
        publish_mosaic(repo, _mosaic(["goes18", "gk2a"], value=9.0), T0,
                       vocabulary=vocabulary)

        stored = _stored(repo)
        assert _times(stored) == [T0, later]
        assert float(stored["u_wind"].values[0].flat[0]) == 9.0
        assert float(stored["u_wind"].values[1].flat[0]) == 2.0
