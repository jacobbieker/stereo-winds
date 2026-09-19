"""Tests for writing global mosaics into an icechunk store."""

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

BASE = Path(__file__).resolve().parent.parent

from stereo_winds.icechunk_output import (  # noqa: E402
    NO_SOURCE,
    align_source_codes,
    icechunk_existing_times,
    mosaic_satellites,
    open_icechunk_repo,
    store_vocabulary,
    write_mosaic_to_icechunk,
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "write_mosaics", BASE / "scripts" / "write_mosaics_to_icechunk.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ingest_script = _load_script()
T0 = datetime(2026, 7, 28)
VARS = ["u_wind", "v_wind", "cloud_top_height", "quality_flag",
        "sigma_u", "sigma_v", "sigma_h"]


def _mosaic(satellites, codes=None, n=4, value=1.0):
    """A mosaic carrying its own satellite vocabulary, as written on disk."""
    data = {v: np.full((n, n), value, np.float32) for v in VARS}
    ds = xr.Dataset(
        {v: (("latitude", "longitude"), data[v]) for v in VARS},
        coords={"latitude": np.linspace(-80, 80, n),
                "longitude": np.linspace(-170, 170, n)},
        attrs={"title": "Global student AMV mosaic", "resolution_m": 10000.0,
               "satellites": list(satellites)},
    )
    if codes is None:
        codes = np.tile(np.arange(len(satellites), dtype=np.int8),
                        (n, n // len(satellites) + 1))[:, :n]
    ds["source_satellite_index"] = (("latitude", "longitude"),
                                    codes.astype(np.int8))
    ds["source_satellite_index"].attrs.update({
        "flag_values": list(range(len(satellites))),
        "flag_meanings": " ".join(satellites),
        "no_source_index": NO_SOURCE,
    })
    return ds


def _write(ds, root, when):
    path = (root / when.strftime("%Y%m%d")
            / f"student_amv_global_{when:%Y%m%dT%H%M}.nc")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def _decode(codes, names):
    out = np.full(codes.shape, "", dtype="U12")
    for code, name in enumerate(names):
        out[codes == code] = name
    return out


class TestDiscovery:
    def test_finds_mosaics_in_time_order(self, tmp_path):
        for hours in (12, 0, 6):
            _write(_mosaic(["goes19"]), tmp_path, T0 + timedelta(hours=hours))
        found = ingest_script.find_mosaics([tmp_path])
        assert [t for t, _ in found] == [T0, T0 + timedelta(hours=6),
                                         T0 + timedelta(hours=12)]

    def test_timestamp_comes_from_the_filename(self, tmp_path):
        path = _write(_mosaic(["goes19"]), tmp_path, T0)
        assert ingest_script.mosaic_time(path) == T0

    def test_unrelated_files_are_ignored(self, tmp_path):
        (tmp_path / "student_amv_goes19_20260728T0000.nc").write_bytes(b"")
        (tmp_path / "notes.txt").write_bytes(b"")
        assert ingest_script.find_mosaics([tmp_path]) == []

    def test_accepts_individual_files(self, tmp_path):
        path = _write(_mosaic(["goes19"]), tmp_path, T0)
        assert ingest_script.find_mosaics([path]) == [(T0, path)]


class TestVocabulary:
    def test_union_in_first_seen_order(self, tmp_path):
        _write(_mosaic(["goes18", "goes19"]), tmp_path, T0)
        _write(_mosaic(["goes18", "goes19", "gk2a"]), tmp_path,
               T0 + timedelta(hours=6))
        _write(_mosaic(["goes19", "himawari9"]), tmp_path,
               T0 + timedelta(hours=12))
        found = ingest_script.find_mosaics([tmp_path])
        assert ingest_script.collect_vocabulary(found) == [
            "goes18", "goes19", "gk2a", "himawari9"]

    def test_starts_from_an_existing_store_vocabulary(self, tmp_path):
        _write(_mosaic(["gk2a"]), tmp_path, T0)
        found = ingest_script.find_mosaics([tmp_path])
        assert ingest_script.collect_vocabulary(
            found, start=["goes18"]) == ["goes18", "gk2a"]

    def test_reads_the_list_from_the_variable(self):
        ds = _mosaic(["goes18", "gk2a"])
        assert mosaic_satellites(ds) == ["goes18", "gk2a"]


class TestAlignSourceCodes:
    def test_codes_are_remapped_to_the_shared_vocabulary(self):
        """Regression: code 2 meant gk2a in one mosaic and himawari9 in another."""
        vocabulary = ["goes18", "goes19", "gk2a"]
        ds = _mosaic(["goes19", "himawari9"],
                     codes=np.array([[0, 1], [1, 0]], np.int8), n=2)
        out = align_source_codes(ds, vocabulary)
        assert vocabulary == ["goes18", "goes19", "gk2a", "himawari9"]
        # goes19 -> 1, himawari9 -> 3
        assert out["source_satellite_index"].values.tolist() == [[1, 3], [3, 1]]

    def test_decoding_is_unchanged_by_the_remap(self):
        vocabulary = ["goes18", "goes19", "gk2a"]
        names = ["goes19", "himawari9"]
        codes = np.array([[0, 1], [1, 0]], np.int8)
        out = align_source_codes(_mosaic(names, codes=codes, n=2), vocabulary)
        assert np.array_equal(
            _decode(codes, names),
            _decode(out["source_satellite_index"].values, vocabulary))

    def test_the_sentinel_survives(self):
        vocabulary = ["goes18"]
        codes = np.array([[NO_SOURCE, 0], [0, NO_SOURCE]], np.int8)
        out = align_source_codes(_mosaic(["goes18"], codes=codes, n=2),
                                 vocabulary)
        assert out["source_satellite_index"].values.tolist() == [
            [NO_SOURCE, 0], [0, NO_SOURCE]]

    def test_existing_codes_keep_their_meaning(self):
        """Names are only appended, so codes already written stay valid."""
        vocabulary = ["goes18", "goes19"]
        align_source_codes(_mosaic(["gk2a"]), vocabulary)
        assert vocabulary[:2] == ["goes18", "goes19"]

    def test_a_mosaic_without_codes_passes_through(self):
        ds = _mosaic(["goes18"]).drop_vars("source_satellite_index")
        assert "source_satellite_index" not in align_source_codes(ds, ["goes18"])


@pytest.fixture
def repo(tmp_path):
    pytest.importorskip("icechunk")
    return open_icechunk_repo(str(tmp_path / "store"))


def _stored(repo):
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False)


class TestIngest:
    def test_one_timestep_per_mosaic(self, tmp_path, repo):
        times = [T0 + timedelta(hours=6 * i) for i in range(3)]
        for when in times:
            _write(_mosaic(["goes18", "goes19"]), tmp_path, when)
        found = ingest_script.find_mosaics([tmp_path])
        written, failed = ingest_script.ingest(found, repo, vocabulary=[])
        assert (written, failed) == (3, [])
        stored = _stored(repo)
        assert stored.sizes["time"] == 3
        assert [pd.Timestamp(v).to_pydatetime()
                for v in stored.time.values] == times

    def test_values_match_the_source(self, tmp_path, repo):
        source = _mosaic(["goes18"], value=3.5)
        _write(source, tmp_path, T0)
        ingest_script.ingest(ingest_script.find_mosaics([tmp_path]), repo,
                             vocabulary=[])
        stored = _stored(repo).isel(time=0)
        for var in VARS:
            assert np.array_equal(stored[var].values, source[var].values,
                                  equal_nan=True)

    def test_resume_skips_what_is_there(self, tmp_path, repo):
        for i in range(2):
            _write(_mosaic(["goes18"]), tmp_path, T0 + timedelta(hours=6 * i))
        found = ingest_script.find_mosaics([tmp_path])
        ingest_script.ingest(found, repo, vocabulary=[])

        _write(_mosaic(["goes18"]), tmp_path, T0 + timedelta(hours=12))
        found = ingest_script.find_mosaics([tmp_path])
        written, _ = ingest_script.ingest(
            found, repo, skip=set(icechunk_existing_times(repo)), vocabulary=[])
        assert written == 1
        assert _stored(repo).sizes["time"] == 3

    def test_provenance_survives_differing_vocabularies(self, tmp_path, repo):
        """The real store mixes 2-, 3- and 4-satellite mosaics."""
        specs = [(["goes18", "goes19"], T0),
                 (["goes18", "goes19", "gk2a"], T0 + timedelta(hours=6)),
                 (["goes19", "himawari9"], T0 + timedelta(hours=12))]
        sources = {}
        for names, when in specs:
            ds = _mosaic(names)
            sources[when] = (names, ds["source_satellite_index"].values.copy())
            _write(ds, tmp_path, when)

        found = ingest_script.find_mosaics([tmp_path])
        vocabulary = ingest_script.collect_vocabulary(found)
        ingest_script.ingest(found, repo, vocabulary=vocabulary)

        stored = _stored(repo)
        store_names = stored["source_satellite_index"].attrs["flag_meanings"].split()
        for when, (names, codes) in sources.items():
            got = stored["source_satellite_index"].sel(
                time=np.datetime64(when, "ns")).values
            assert np.array_equal(_decode(codes, names),
                                  _decode(got, store_names)), when

    def test_vocabulary_grows_without_invalidating_earlier_steps(
            self, tmp_path, repo):
        first = _mosaic(["goes18", "goes19"])
        _write(first, tmp_path, T0)
        found = ingest_script.find_mosaics([tmp_path])
        ingest_script.ingest(found, repo,
                             vocabulary=ingest_script.collect_vocabulary(found))
        assert store_vocabulary(repo) == ["goes18", "goes19"]

        later = _mosaic(["goes18", "himawari9"])
        path = _write(later, tmp_path, T0 + timedelta(hours=6))
        found = ingest_script.find_mosaics([path])
        vocabulary = ingest_script.collect_vocabulary(
            found, store_vocabulary(repo))
        ingest_script.ingest(found, repo, vocabulary=vocabulary)

        assert store_vocabulary(repo) == ["goes18", "goes19", "himawari9"]
        stored = _stored(repo)
        names = stored["source_satellite_index"].attrs["flag_meanings"].split()
        assert np.array_equal(
            _decode(first["source_satellite_index"].values,
                    ["goes18", "goes19"]),
            _decode(stored["source_satellite_index"].isel(time=0).values, names))

    def test_an_unreadable_mosaic_does_not_stop_the_rest(self, tmp_path, repo):
        _write(_mosaic(["goes18"]), tmp_path, T0)
        bad = (tmp_path / "20260728"
               / "student_amv_global_20260728T0600.nc")
        bad.write_bytes(b"not a netcdf")
        _write(_mosaic(["goes18"]), tmp_path, T0 + timedelta(hours=12))

        found = ingest_script.find_mosaics([tmp_path])
        written, failed = ingest_script.ingest(found, repo, vocabulary=[])
        assert written == 2
        assert failed == [T0 + timedelta(hours=6)]
