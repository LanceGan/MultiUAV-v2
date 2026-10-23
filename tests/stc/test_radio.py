"""Task 3: deterministic per-sector radio field with explicit map seed."""

import numpy as np
import pytest

from segment.stc.radio import PerSectorRadioField, configure_radio_world
from segment.stc.scenario import load_stc_scenario


def test_same_map_seed_reproduces_world_hash():
    h1 = configure_radio_world(4000.0, 4, map_seed=0)
    h2 = configure_radio_world(4000.0, 4, map_seed=0)
    assert h1 == h2


def test_different_map_seed_changes_world_hash():
    h1 = configure_radio_world(4000.0, 4, map_seed=0)
    h2 = configure_radio_world(4000.0, 4, map_seed=1)
    assert h1 != h2


def test_radio_configuration_does_not_mutate_numpy_global_rng():
    np.random.seed(12345)
    before = np.random.get_state()[1].copy()
    configure_radio_world(4000.0, 4, map_seed=0)
    after = np.random.get_state()[1].copy()
    assert np.array_equal(before, after)


def test_field_sample_returns_per_sector_arrays():
    sc = load_stc_scenario(2, task_seed=42, map_seed=0)
    field = PerSectorRadioField.from_scenario(sc, spacing_m=1000.0)
    sample = field.sample(np.array([2000.0, 2000.0]))
    assert sample.a2g_efficiency.shape == (field.n_sectors,)
    assert sample.g2a_outage.shape == (field.n_sectors,)


def test_field_rejects_out_of_region():
    sc = load_stc_scenario(2, task_seed=42, map_seed=0)
    field = PerSectorRadioField.from_scenario(sc, spacing_m=1000.0)
    with pytest.raises(ValueError):
        field.sample(np.array([99999.0, 99999.0]))
