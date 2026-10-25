"""Legacy-to-SI scenario conversion (Task 2).

``load_stc_scenario`` reads the legacy scenario (100-m coordinates, Mbit payloads)
and returns an SI ``STCScenario`` (metres, bits).  Unit conversion and seed
separation live only here and in ``radio.py``.
"""

import json
import os
from pathlib import Path

import numpy as np

from common.config import get_data_sizes, get_scenario
from segment.stc.types import STCScenario, ScenarioLimits

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results" / "datas"


def _load_limits(config_path):
    if config_path is None:
        config_path = ROOT / "configs" / "stc" / "base.json"
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    return ScenarioLimits(**data)


def load_stc_scenario(uav_num, *, task_seed, map_seed, config_path=None):
    """Load a legacy scenario and convert it to SI units.

    ``task_seed`` selects the per-point payload sizes; ``map_seed`` is stored
    for the radio-field adapter (Task 3) and does not affect the tasks here.
    """
    cfg = get_scenario(uav_num)
    path = DATA_DIR / ("Users_%d.txt" % int(cfg["user_num"]))
    points = np.loadtxt(path)[:, :2]
    data_sizes = get_data_sizes(int(cfg["user_num"]), seed=int(task_seed))

    positions_m = tuple((float(p[0]) * 100.0, float(p[1]) * 100.0) for p in points)
    data_bits = tuple(float(d) * 1e6 for d in data_sizes)
    depot_m = (float(cfg["ini_loc"][0]) * 100.0, float(cfg["ini_loc"][1]) * 100.0)
    terminal_m = (float(cfg["end_loc"][0]) * 100.0, float(cfg["end_loc"][1]) * 100.0)

    return STCScenario(
        n_uav=int(uav_num),
        n_tasks=len(points),
        task_positions_m=positions_m,
        task_data_bits=data_bits,
        depot_m=depot_m,
        terminal_m=terminal_m,
        limits=_load_limits(config_path),
        length_m=float(cfg["length"]) * 100.0,
        width_m=float(cfg["width"]) * 100.0,
        n_bs=len(cfg["bs_loc"]),
        task_seed=int(task_seed),
        map_seed=int(map_seed),
    )
