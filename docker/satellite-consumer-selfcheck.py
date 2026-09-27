"""Fail the image build if it cannot read what it exists to read.

The upstream Dockerfile installs from uv.lock, which carries none of
netcdf4, h5netcdf or pyspectral, so satpy silently ends up with no
reader for FCI or SEVIRI.  Nothing complains until a run has downloaded
a whole window and has to open it, which is an expensive way to find
out.  Assert it at build time instead.
"""

import sys

# Moved in satpy 0.60; the old path still works but warns.
try:
    from satpy.readers.core.config import available_readers
except ImportError:  # pragma: no cover - older satpy
    from satpy.readers import available_readers

#: fci_l1c_nc reads MTG (odegree-12), seviri_l1b_native reads MSG and
#: IODC (odegree, iodc, rss).
REQUIRED = {"fci_l1c_nc", "seviri_l1b_native"}

readers = set(available_readers())
missing = sorted(REQUIRED - readers)
if missing:
    sys.exit(
        f"satpy is missing readers {missing}; the image cannot process "
        f"the satellites it is built for ({len(readers)} readers present)"
    )

# netCDF4 importing is not the same as netCDF4 working: the PyPI wheels
# import cleanly and then fail inside dask with "NetCDF: HDF error".
import netCDF4  # noqa: E402

print(f"satpy readers OK ({len(readers)} present), netCDF4 {netCDF4.__version__}")
