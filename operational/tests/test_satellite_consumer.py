"""Tests for the satellite-consumer resource's container interface."""

import datetime as dt

import pytest

from operational.satellite_consumer import (
    CONSUMER_SATELLITES,
    SatelliteConsumerResource,
    consumer_satellite,
    window_for,
)

T0 = dt.datetime(2026, 9, 20, 0, 0)


@pytest.fixture
def resource():
    # Explicit values stand in for the EnvVar defaults, which only
    # resolve inside a Dagster run.
    return SatelliteConsumerResource(
        eumetsat_key="eum-key",
        eumetsat_secret="eum-secret",
        aws_access_key_id="aws-key",
        aws_secret_access_key="aws-secret",
    )


class TestSatelliteRegistry:
    def test_covers_the_three_eumetsat_satellites(self):
        assert set(CONSUMER_SATELLITES) == {"odegree-12", "odegree", "iodc"}

    def test_keys_are_the_consumers_own_names(self):
        """SATCONS_SATELLITE takes these, not the ring's satellite ids."""
        for key, sat in CONSUMER_SATELLITES.items():
            assert sat.key == key
        assert consumer_satellite("odegree-12").ring_id == "mtg-i1"
        assert consumer_satellite("iodc").ring_id == "msg-iodc"

    def test_unknown_satellite_names_the_known_ones(self):
        with pytest.raises(KeyError, match="odegree-12"):
            consumer_satellite("mtg-i1")

    def test_stores_match_what_the_readers_discover(self):
        """A consumer run must top up the store a retrieval later reads."""
        assert consumer_satellite("odegree-12").store.startswith("geo/mtg_")
        assert consumer_satellite("iodc").store.startswith("geo/iodc_")

    def test_resolutions_are_the_tier_the_wind_bands_live_in(self):
        # Not FCI's finest: the winds use the IR/WV bands and only
        # mtg_2000m carries all eight, so filling mtg_1000m would leave
        # the retrieval exactly as short of MTG as before.
        assert consumer_satellite("odegree-12").resolution_m == 2000
        assert consumer_satellite("odegree").resolution_m == 3000  # SEVIRI
        assert consumer_satellite("iodc").resolution_m == 3000

    def test_mtg_targets_the_store_the_retrieval_reads(self):
        from stereo_winds.readers.mtg import MTG

        reader = MTG("mtg-i1")
        wanted = reader._store_prefix(reader.band_resolution["ir_105"])
        assert consumer_satellite("odegree-12").store == wanted


class TestWindowEnv:
    def test_matches_a_known_good_invocation(self, resource):
        """The env a working manual run used, minus the credentials."""
        sat = consumer_satellite("odegree-12")
        env = resource.window_env(sat, T0, T0 + dt.timedelta(minutes=10))
        assert env["SATCONS_SATELLITE"] == "odegree-12"
        assert env["SATCONS_RESOLUTION"] == "2000"
        assert env["SATCONS_ICECHUNK"] == "True"
        assert env["SATCONS_START_TIMESTAMP"] == "2026-09-20T00:00:00Z"
        assert env["SATCONS_END_TIMESTAMP"] == "2026-09-20T00:10:00Z"
        assert env["SATCONS_ZARR_PATH"] == (
            "s3://us-west-2.opendata.source.coop/bkr/geo/mtg_2000m.icechunk"
        )

    def test_carries_no_credentials(self, resource):
        """Windows get logged; secrets must not ride along with them."""
        env = resource.window_env(consumer_satellite("iodc"), T0, T0 + dt.timedelta(minutes=15))
        assert not [k for k in env if "KEY" in k or "SECRET" in k]
        assert "eum-secret" not in "".join(env.values())

    def test_credentials_are_separate_and_complete(self, resource):
        creds = resource.credential_env()
        assert creds["EUMETSAT_CONSUMER_KEY"] == "eum-key"
        assert creds["AWS_SECRET_ACCESS_KEY"] == "aws-secret"
        assert creds["AWS_DEFAULT_REGION"] == "us-west-2"

    def test_backwards_window_is_refused(self, resource):
        with pytest.raises(ValueError, match="not after"):
            resource.window_env(consumer_satellite("iodc"), T0, T0 - dt.timedelta(minutes=15))

    def test_aware_timestamps_convert_rather_than_truncate(self, resource):
        """Dropping a +02:00 offset would shift the window two hours."""
        aware = dt.datetime(2026, 9, 20, 2, 0, tzinfo=dt.timezone(dt.timedelta(hours=2)))
        env = resource.window_env(
            consumer_satellite("odegree-12"), aware, aware + dt.timedelta(minutes=10)
        )
        assert env["SATCONS_START_TIMESTAMP"] == "2026-09-20T00:00:00Z"

    def test_raw_path_is_under_the_workdir(self, resource):
        env = resource.window_env(consumer_satellite("odegree"), T0, T0 + dt.timedelta(minutes=15))
        assert env["SATCONS_RAW_PATH"].startswith(env["SATCONS_WORKDIR"])
        # Raw files are scratch; keeping them fills the volume.
        assert env["SATCONS_KEEP_RAW"] == "False"

    def test_store_url_joins_without_doubling_slashes(self):
        r = SatelliteConsumerResource(
            bucket_url="s3://bucket/prefix/",
            eumetsat_key="k",
            eumetsat_secret="s",
            aws_access_key_id="a",
            aws_secret_access_key="b",
        )
        assert r.store_url(consumer_satellite("iodc")) == (
            "s3://bucket/prefix/geo/iodc_3000m.icechunk"
        )


class TestWindowFor:
    def test_one_cycle_is_the_satellites_cadence(self):
        sat = consumer_satellite("odegree-12")
        start, end = window_for(sat, T0)
        assert end - start == dt.timedelta(minutes=10)

    def test_seviri_cadence_differs_from_fci(self):
        _, end = window_for(consumer_satellite("iodc"), T0)
        assert end - T0 == dt.timedelta(minutes=15)

    def test_multiple_cycles(self):
        _, end = window_for(consumer_satellite("odegree-12"), T0, cycles=6)
        assert end - T0 == dt.timedelta(hours=1)

    def test_zero_cycles_is_refused(self):
        with pytest.raises(ValueError, match="positive"):
            window_for(consumer_satellite("iodc"), T0, cycles=0)


class TestCredentialResolution:
    """A missing ingest credential must not break the whole code location."""

    def test_constructs_with_nothing_set(self, monkeypatch):
        for var in (
            "EUMETSAT_CONSUMER_KEY",
            "EUMETSAT_CONSUMER_SECRET",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        # Definition-time construction must not raise: an EnvVar default
        # would have, taking the retrieval assets down with it.
        SatelliteConsumerResource()

    def test_reads_the_environment_when_unset(self, monkeypatch):
        monkeypatch.setenv("EUMETSAT_CONSUMER_KEY", "from-env")
        monkeypatch.setenv("EUMETSAT_CONSUMER_SECRET", "s")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "a")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "b")
        creds = SatelliteConsumerResource().credential_env()
        assert creds["EUMETSAT_CONSUMER_KEY"] == "from-env"

    def test_explicit_config_wins(self, monkeypatch):
        monkeypatch.setenv("EUMETSAT_CONSUMER_KEY", "from-env")
        monkeypatch.setenv("EUMETSAT_CONSUMER_SECRET", "s")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "a")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "b")
        r = SatelliteConsumerResource(eumetsat_key="explicit")
        assert r.credential_env()["EUMETSAT_CONSUMER_KEY"] == "explicit"

    def test_missing_names_all_of_them_at_once(self, monkeypatch):
        for var in (
            "EUMETSAT_CONSUMER_KEY",
            "EUMETSAT_CONSUMER_SECRET",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(RuntimeError) as exc:
            SatelliteConsumerResource().credential_env()
        message = str(exc.value)
        # One container start and EUMETSAT round trip per missing name is
        # an expensive way to discover them one at a time.
        for var in ("EUMETSAT_CONSUMER_KEY", "AWS_SECRET_ACCESS_KEY"):
            assert var in message
