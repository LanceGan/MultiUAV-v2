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

# NOTE (units): scenario ``v_max`` is in 100m-units/s, so ``v_max=0.5`` == 50 m/s
# physical (the single-UAV paper's 20 m/s would be ``v_max=0.2``). Flag for a
# future sensitivity analysis — do NOT change ``v_max`` now (would invalidate
# existing smoke results).


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
    """Return per-point offload volume in Mbit (uniform 10–30 Mbit).

    Unit convention: data sizes are Mbit, so the offload time is
    ``tau = data_buffer / rate`` in seconds with no 8x factor (``rate`` is Mbps).
    10–30 Mbit ≈ a few inspection photos / a short video clip per point, which
    keeps the offload delay at B=1 MHz in the tens-of-seconds range (was 100–300
    Mbit -> hundreds of seconds, the "data-rate mismatch" fixed by experiment 5).
    """
    rng = np.random.RandomState(seed)
    return rng.uniform(10.0, 30.0, user_num)  # Mbit
