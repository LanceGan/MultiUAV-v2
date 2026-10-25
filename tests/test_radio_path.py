import numpy as np
from sequence import radio_path
from common.config import get_scenario


def test_edge_cost_is_positive_finite():
    # communication-aware cost is NOT monotonic with distance (rural terrain makes
    # outage/delay vary per edge); only assert both costs are positive and finite.
    cfg = get_scenario(3)
    a = np.array([5.0, 5.0]); b = np.array([6.0, 5.0]); c = np.array([9.0, 5.0])
    cab = radio_path.edge_cost(a, b, 100.0, cfg)["cost"]
    cac = radio_path.edge_cost(a, c, 100.0, cfg)["cost"]
    assert cab > 0 and cac > 0 and np.isfinite(cab) and np.isfinite(cac)


def test_edge_cost_penalizes_delay_violation():
    cfg = get_scenario(3)
    a = np.array([5.0, 5.0]); b = np.array([6.0, 5.0])
    big = radio_path.edge_cost(a, b, 1e6, cfg)["cost"]   # 数据量大→时延违约
    small = radio_path.edge_cost(a, b, 10.0, cfg)["cost"]
    assert big > small
