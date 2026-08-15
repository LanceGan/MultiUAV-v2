import numpy as np

from sequence import ga_eqtsp, baselines, radio_path
from common.config import get_scenario


def test_ga_eqtsp_visits_all_points_once():
    cfg = get_scenario(2)
    pts = np.random.RandomState(0).uniform(0, 40, (8, 2))
    ds = np.ones(8) * 100.0
    ordered, order, cost = ga_eqtsp.ga_eqtsp(
        pts, np.array([2., 2.]), np.array([38., 38.]), ds, cfg)
    assert len(set(order)) == 8  # 每点访问一次
    assert len(order) == 8
    assert np.isfinite(cost)
    assert ordered.shape == (10, 2)


def test_ga_eqtsp_beats_euclidean_on_comm_cost():
    # 通信感知边权下，GA_EQTSP 的通信代价 <= 纯欧氏 GA 的通信代价
    cfg = get_scenario(2)
    pts = np.random.RandomState(1).uniform(0, 40, (6, 2))
    ds = np.ones(6) * 300.0

    def comm_cost(order):
        # edge_cost 内含瑞利快衰落（np.random.randn），故每次评估前固定种子，
        # 使同一 order 的通信代价可复现，避免噪声翻转 <= 比较。
        np.random.seed(0)
        return sum(radio_path.edge_cost(
            pts[order[i]], pts[order[i + 1]], ds[order[i]], cfg)["cost"]
            for i in range(len(order) - 1))

    _, o_eq, _ = ga_eqtsp.ga_eqtsp(
        pts, np.array([2., 2.]), np.array([38., 38.]), ds, cfg)
    _, o_ga, _ = baselines.ga_euclidean(
        pts, np.array([2., 2.]), np.array([38., 38.]), ds, cfg)
    assert comm_cost(o_eq) <= comm_cost(o_ga)


def test_baselines_visit_all_points_once():
    cfg = get_scenario(2)
    pts = np.random.RandomState(2).uniform(0, 40, (5, 2))
    ds = np.ones(5) * 150.0
    ini = np.array([2., 2.])
    end = np.array([38., 38.])
    for fn in (baselines.ga_euclidean, baselines.pso, baselines.aco,
               baselines.nn_2opt):
        ordered, order, cost = fn(pts, ini, end, ds, cfg)
        assert sorted(order) == list(range(5))
        assert np.isfinite(cost)
        assert ordered.shape == (7, 2)
        assert np.allclose(ordered[0], ini)
        assert np.allclose(ordered[-1], end)
