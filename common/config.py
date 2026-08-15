"""Scenario configuration (Task 11).

The per-scenario parameters (``user_num`` / ``length`` / ``width`` / ``v_max`` /
``delta_t`` / ``tau_max`` / ``T_out_max`` / ``lambda_*`` / ``bs_loc`` /
``ini_loc`` / ``end_loc``) live in ``configs/scenario_uav<N>.json`` so they can
be edited / re-run without touching code.  ``get_scenario(uav_num)`` loads the
corresponding JSON and returns the *same* dict interface as before (numpy
arrays for the three location fields, ``uav_num`` / ``seed`` added from the
call), so existing callers and tests are unaffected.

``ENERGY_PARAMS`` (rotary-wing power model) and ``get_data_sizes`` (per-point
offload volumes) remain here in ``common.config``.
"""

import json
import os

import numpy as np

ENERGY_PARAMS = dict(P0=79.8563, Pi=88.6279, U_tip=120.0, v0=4.03,
                     d0=0.6, rho=1.225, s=0.05, A=0.503)

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, "configs")

# Fields serialised as JSON lists but returned as numpy arrays (matching the
# original hardcoded ``get_scenario`` interface).
_ARRAY_KEYS = ("bs_loc", "ini_loc", "end_loc")


def _load_scenario_json(uav_num):
    """Load ``configs/scenario_uav<N>.json`` and convert location lists to arrays."""
    path = os.path.join(_CONFIG_DIR, "scenario_uav%d.json" % int(uav_num))
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in _ARRAY_KEYS:
        data[key] = np.asarray(data[key], dtype=float)
    return data


def get_scenario(uav_num, seed=42):
    """Return the scenario configuration dict for ``uav_num`` UAVs.

    Interface is unchanged: keys ``user_num`` / ``length`` / ``width`` /
    ``v_max`` / ``delta_t`` / ``tau_max`` / ``T_out_max`` / ``lambda_E`` /
    ``lambda_out`` / ``lambda_tau`` / ``bs_loc`` / ``ini_loc`` / ``end_loc``
    (numpy arrays) plus ``uav_num`` and ``seed``.
    """
    sc = _load_scenario_json(uav_num)
    sc["uav_num"] = int(uav_num)
    sc["seed"] = int(seed)
    return sc


def get_data_sizes(user_num, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(100.0, 300.0, user_num)  # MB，对齐单机论文测试的 100~300MB 量级
