"""Deterministic per-sector radio field with explicit map seed (Task 3).

``configure_radio_world`` configures both legacy radio maps for a square region
and returns a content hash of the resulting world (terrain + base stations).
``PerSectorRadioField`` samples the per-sector A2G spectral efficiency and G2A
outage over a grid once, then serves bilinear-interpolated values in SI units.
"""

import hashlib

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from radio_map import radio_map_A2G, radio_map_G2A
from segment.stc.types import RadioSample

SIR_THRESHOLD = 3.0  # must match radio_map_A2G.SIR_THRESHOLD


def configure_radio_world(length_m, n_bs, map_seed, uplink_power_w=None):
    """Configure both maps for a ``length_m`` x ``length_m`` region, return world hash.

    ``uplink_power_w`` (Watt) is applied to the A2G uplink map so a scenario's
    uplink power actually enters the radio propagation, and is folded into the
    world hash so two otherwise-identical worlds differing only in uplink power
    are not treated as interchangeable.
    """
    d_km = float(length_m) / 1000.0
    radio_map_A2G.set_region(d_km, int(n_bs), int(map_seed))
    radio_map_G2A.set_region(d_km, int(n_bs), int(map_seed))
    if uplink_power_w is not None:
        radio_map_A2G.set_uplink_power(float(uplink_power_w))
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(radio_map_A2G.HeightMapMatrix).tobytes())
    h.update(np.ascontiguousarray(radio_map_A2G.BS_loc).tobytes())
    h.update(repr(float(radio_map_A2G.PB)).encode("utf-8"))
    return h.hexdigest()


class PerSectorRadioField:
    """Cached per-sector A2G/G2A field with bilinear interpolation (SI units)."""

    def __init__(self, scenario, spacing_m=100.0):
        self.n_sectors = scenario.n_sectors
        self.length_m = scenario.length_m
        self.width_m = scenario.width_m
        self.world_hash = configure_radio_world(
            max(scenario.length_m, scenario.width_m), scenario.n_bs, scenario.map_seed,
            uplink_power_w=scenario.limits.uplink_power_w)

        xs = np.arange(0.0, scenario.length_m + 1e-9, spacing_m)
        ys = np.arange(0.0, scenario.width_m + 1e-9, spacing_m)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        h_km = scenario.limits.uav_height_m / 1000.0

        a2g = np.zeros((len(xs), len(ys), self.n_sectors))
        g2a = np.zeros((len(xs), len(ys), self.n_sectors))
        for i in range(len(xs)):
            for j in range(len(ys)):
                loc = np.array([gx[i, j] / 1000.0, gy[i, j] / 1000.0, h_km])
                _, a2g_sinr = radio_map_A2G.getPointOutageMatrix(loc, SIR_THRESHOLD)
                g2a_out, _ = radio_map_G2A.getPointOutageMatrix(loc, SIR_THRESHOLD)
                # flatten (numBS, Sec_num) -> (n_sectors,); SINR dB -> efficiency
                a2g[i, j] = np.log2(1.0 + 10.0 ** (np.asarray(a2g_sinr).ravel() / 10.0))
                g2a[i, j] = np.asarray(g2a_out).ravel()

        self._a2g = [RegularGridInterpolator((xs, ys), a2g[:, :, m])
                     for m in range(self.n_sectors)]
        self._g2a = [RegularGridInterpolator((xs, ys), g2a[:, :, m])
                     for m in range(self.n_sectors)]

    @classmethod
    def from_scenario(cls, scenario, spacing_m=100.0):
        return cls(scenario, spacing_m=spacing_m)

    def sample(self, position_m):
        """Return a ``RadioSample`` with per-sector A2G efficiency and G2A outage.

        Raises ``ValueError`` for a position outside the region.
        """
        pos = np.asarray(position_m, dtype=float).reshape(1, -1)
        a2g = np.array([f(pos)[0] for f in self._a2g])
        g2a = np.array([f(pos)[0] for f in self._g2a])
        return RadioSample(a2g_efficiency=a2g, g2a_outage=g2a)
