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
    # GA_EQTSP 显式最小化 comm_dis_mat，故其输出路线在同一目标下的代价不应高于
    # 纯欧氏 GA 的路线（欧氏路线只是 comm_dis_mat 上的一个可行解）。
    # 异构数据量（500 Mbit 点）放大了时延项，确保通信目标与欧氏目标有实质差异。
    # 注：comm_dis_mat 已与 edge_cost 逐边一致（见 test_comm_dis_mat_consistent...），
    # 故此断言等价于"报告的 comm_cost 不劣于欧氏基线"。
    cfg = get_scenario(2)
    pts = np.random.RandomState(1).uniform(0, 40, (6, 2))
    ds = np.array([50., 50., 50., 500., 500., 500.])
    ini = np.array([2., 2.])
    end = np.array([38., 38.])

    full_data = np.vstack([ini, pts, end])
    mat = ga_eqtsp.comm_dis_mat(full_data, ds, cfg)

    def mat_cost(order):
        path = [0] + [i + 1 for i in order] + [len(full_data) - 1]
        return sum(mat[path[i]][path[i + 1]] for i in range(len(path) - 1))

    _, o_eq, _ = ga_eqtsp.ga_eqtsp(pts, ini, end, ds, cfg, iteration=300)
    _, o_ga, _ = baselines.ga_euclidean(pts, ini, end, ds, cfg)
    assert mat_cost(o_eq) <= mat_cost(o_ga) * (1.0 + 1e-9)


def test_comm_dis_mat_consistent_with_edge_cost():
    # comm_dis_mat 的每条有向边代价必须等于 edge_cost（含 flight_time 时延因子）。
    # 回归：comm_dis_mat 曾漏乘 flight_time，GA 优化时用的是无量纲时延 ds/C_ij
    # 而非秒（ds·flight_time/C_ij），与 edge_cost 不一致。
    cfg = get_scenario(2)
    pts = np.random.RandomState(3).uniform(0, 40, (5, 2))
    ds = np.array([50., 200., 300., 500., 1000.])  # 1000 Mbit 保证触发时延违约
    ini = np.array([2., 2.])
    end = np.array([38., 38.])
    full_data = np.vstack([ini, pts, end])

    mat = ga_eqtsp.comm_dis_mat(full_data, ds, cfg)

    n = full_data.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ds_i = ds[i - 1] if 0 < i < n - 1 else 0.0
            expected = radio_path.edge_cost(
                full_data[i][:2], full_data[j][:2], ds_i, cfg)["cost"]
            assert abs(mat[i][j] - expected) <= 1e-6 * (1.0 + abs(expected)), (
                f"comm_dis_mat[{i}][{j}] 与 edge_cost 不一致："
                f"{mat[i][j]:.4f} != {expected:.4f}")


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
