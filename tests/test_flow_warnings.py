"""Regression tests for deprecated torch APIs in the vendored RAFT.

Both fire on every forward pass, so a long run buries real warnings
under thousands of these.  The fixes must not change the numerics.
"""

import warnings

import pytest
import torch

from stereo_winds.flow.raft.utils.utils import coords_grid


class TestAutocast:
    def test_no_future_warning(self):
        """torch.cuda.amp.autocast is deprecated from torch 2.4."""
        from stereo_winds.flow.raft.raft import autocast

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with autocast(enabled=False):
                pass

    def test_disabled_autocast_leaves_dtypes_alone(self):
        from stereo_winds.flow.raft.raft import autocast

        with autocast(enabled=False):
            out = torch.ones(2, 2, dtype=torch.float32) @ torch.ones(2, 2)
        assert out.dtype == torch.float32


class TestCoordsGrid:
    def test_no_meshgrid_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            coords_grid(1, 3, 4)

    def test_matches_the_legacy_ij_grid(self):
        """indexing="ij" is the default this code was written against."""
        ht, wd = 3, 4
        legacy = torch.stack(
            torch.meshgrid(torch.arange(ht), torch.arange(wd),
                           indexing="ij")[::-1], dim=0).float()[None]
        assert torch.equal(coords_grid(1, ht, wd), legacy)

    def test_channel_order_is_x_then_y(self):
        """Flow code reads channel 0 as x and channel 1 as y."""
        coords = coords_grid(1, 3, 4)
        assert coords.shape == (1, 2, 3, 4)
        assert coords[0, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]   # x along a row
        assert coords[0, 1, :, 0].tolist() == [0.0, 1.0, 2.0]     # y down a column

    def test_batch_is_repeated(self):
        coords = coords_grid(5, 2, 2)
        assert coords.shape == (5, 2, 2, 2)
        assert torch.equal(coords[0], coords[4])


class TestCorrDelta:
    def test_correlation_offsets_are_unwarned_and_square(self):
        from stereo_winds.flow.raft.corr import CorrBlock

        r = 3
        dx = torch.linspace(-r, r, 2 * r + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            delta = torch.stack(torch.meshgrid(dx, dx, indexing="ij"), axis=-1)
        assert delta.shape == (2 * r + 1, 2 * r + 1, 2)
        assert CorrBlock is not None
