import numpy as np

from sequence import radio_path
from common.config import get_scenario
from experiment import _common


def test_path_cost_pairs_data_by_visit_order():
    # path_cost 必须按访问序把数据量配到边：访问的第 k 个点携带 data_sizes[order[k]]。
    # 回归：曾按自然序 data_sizes[i-1] 配对，非自然访问序时把数据量配到错误的点。
    cfg = get_scenario(2)
    ini = np.array([2., 2.])
    end = np.array([38., 38.])
    pts = np.random.RandomState(0).uniform(0, 40, (3, 2))
    ds = np.array([50., 100., 1e5])  # 异构数据量，1e5 Mbit 保证触发时延违约
    order = [2, 0, 1]  # 非自然访问序：先访问数据量 400 的点
    ordered = np.vstack([ini, pts[order], end])  # [ini, p2, p0, p1, end]

    def comm_with(ds_seq):
        # 手动逐边求和：ds_seq 须与 ordered 的访问序对齐
        n = ordered.shape[0]
        c = 0.0
        for i in range(n - 1):
            d = ds_seq[i - 1] if 0 < i < n - 1 else 0.0
            c += radio_path.edge_cost(
                ordered[i][:2], ordered[i + 1][:2], d, cfg)["cost"]
        return c

    correct = comm_with(ds[order])   # 正确：按访问序配数据
    wrong = comm_with(ds)            # buggy：按自然序配数据
    assert abs(correct - wrong) > 1e-6  # 异构数据量下两种配对确实不同

    got = _common.path_cost(ordered, ds, order, cfg, 0)[1]
    assert abs(got - correct) < 1e-9
