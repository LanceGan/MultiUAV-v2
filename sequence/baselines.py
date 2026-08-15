"""Layer-2 序列决策基线（纯欧氏距离，无通信感知）。

提供与 ``ga_eqtsp`` 相同签名的四个基线：GA（欧氏）、PSO、ACO、最近邻+2-opt。
均返回 ``(ordered_points, order_indices, cost)``，其中 cost 为全程欧氏路径长度。
"""

import math
import random

import numpy as np

from sequence.ga_eqtsp import GA, euclidean_dis_mat, _prepare, _order_and_coords


def _seed(cfg):
    np.random.seed(int(cfg["seed"]))
    random.seed(int(cfg["seed"]))


def ga_euclidean(cluster_points, ini_loc, end_loc, data_sizes, cfg,
                 num_total=25, iteration=200):
    """纯欧氏距离的 GA 基线。"""
    _seed(cfg)
    full_data, _ = _prepare(cluster_points, ini_loc, end_loc, data_sizes)
    dis_mat = euclidean_dis_mat(full_data)
    model = GA(num_city=full_data.shape[0], num_total=num_total,
               iteration=iteration, data=full_data.copy(), dis_mat=dis_mat)
    coords, cost, indices = model.run()
    return _order_and_coords(full_data, indices, cost)


class _PSO(object):
    """迁移自参考实现的 PSO（离散粒子群，起点/终点锁定）。"""

    def __init__(self, num_city, data, dis_mat, num=60, iter_max=200):
        self.iter_max = iter_max
        self.num = num
        self.num_city = num_city
        self.location = data
        self.dis_mat = dis_mat
        self.particals = self.greedy_init(dis_mat, num_total=self.num,
                                          num_city=num_city)
        self.lenths = self.compute_paths(self.particals)
        init_l = min(self.lenths)
        init_index = self.lenths.index(init_l)
        init_path = self.particals[init_index]
        self.local_best = self.particals
        self.local_best_len = self.lenths
        self.global_best = init_path
        self.global_best_len = init_l

    def greedy_init(self, dis_mat, num_total, num_city):
        result = []
        start, end = 0, num_city - 1
        for _ in range(num_total):
            rest = [x for x in range(num_city) if x != start and x != end]
            current = start
            result_one = [start]
            while rest:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result_one.append(end)
            result.append(result_one)
        return result

    def compute_pathlen(self, path):
        result = 0.0
        for i in range(len(path) - 1):
            result += self.dis_mat[path[i]][path[i + 1]]
        return result

    def compute_paths(self, paths):
        return [self.compute_pathlen(one) for one in paths]

    def eval_particals(self):
        for i, l in enumerate(self.lenths):
            if l < self.local_best_len[i]:
                self.local_best_len[i] = l
                self.local_best[i] = self.particals[i]
            if l < self.global_best_len:
                self.global_best_len = l
                self.global_best = self.particals[i]

    def cross(self, cur, best):
        start, end = cur[0], cur[-1]
        middle = cur[1:-1]
        if len(middle) < 2:
            return list(cur), self.compute_pathlen(cur)
        l = list(range(len(middle)))
        x, y = np.random.choice(l, 2, replace=False)
        x, y = min(x, y), max(x, y)
        cross_part = best[1:-1][x:y]
        tmp = [t for t in middle if t not in cross_part]
        one1 = [start] + tmp + cross_part + [end]
        one2 = [start] + cross_part + tmp + [end]
        l1 = self.compute_pathlen(one1)
        l2 = self.compute_pathlen(one2)
        return (one1, l1) if l1 < l2 else (one2, l2)

    def mutate(self, one):
        start, end = one[0], one[-1]
        middle = one[1:-1]
        if len(middle) < 2:
            return list(one), self.compute_pathlen(one)
        l = list(range(len(middle)))
        x, y = np.random.choice(l, 2, replace=False)
        middle[x], middle[y] = middle[y], middle[x]
        mutated = [start] + middle + [end]
        return mutated, self.compute_pathlen(mutated)

    def pso(self):
        for _ in range(self.iter_max):
            for i in range(self.num):
                one = self.particals[i]
                l = self.lenths[i]
                for target in (self.local_best[i], self.global_best):
                    cand, cand_l = self.cross(one, target)
                    if cand_l < l or np.random.rand() < 0.1:
                        one, l = cand, cand_l
                cand, cand_l = self.mutate(one)
                if cand_l < l or np.random.rand() < 0.1:
                    one, l = cand, cand_l
                self.particals[i] = one
                self.lenths[i] = l
            self.eval_particals()
        return self.global_best_len, self.global_best

    def run(self):
        best_length, best_path = self.pso()
        return self.location[best_path], best_length, best_path


def pso(cluster_points, ini_loc, end_loc, data_sizes, cfg,
        num=60, iter_max=200):
    """纯欧氏距离的 PSO 基线。"""
    _seed(cfg)
    full_data, _ = _prepare(cluster_points, ini_loc, end_loc, data_sizes)
    dis_mat = euclidean_dis_mat(full_data)
    model = _PSO(num_city=full_data.shape[0], data=full_data.copy(),
                 dis_mat=dis_mat, num=num, iter_max=iter_max)
    coords, cost, indices = model.run()
    return _order_and_coords(full_data, indices, cost)


class _ACO(object):
    """迁移自参考实现的 ACO（蚁群，起点/终点锁定）。"""

    def __init__(self, num_city, data, dis_mat, m=50, iter_max=200,
                 start_node=0, end_node=None):
        self.m = m
        self.alpha = 1
        self.beta = 5
        self.rho = 0.1
        self.Q = 1
        self.num_city = num_city
        self.location = data
        self.dis_mat = dis_mat
        self.start_node = start_node
        self.end_node = end_node if end_node is not None else num_city - 1
        self.iter_max = iter_max
        # 初始信息素取正数，使首轮由启发式（Eta）主导，避免参考实现 Tau=0 导致
        # 首轮概率全 0 / NaN 的问题。
        self.Tau = np.full((num_city, num_city), 1.0)
        self.Eta = 10.0 / np.where(dis_mat > 0, dis_mat, np.inf)
        self.Table = [[0 for _ in range(num_city)] for _ in range(self.m)]
        self.paths = None

    def rand_choose(self, p):
        x = np.random.rand()
        for i, t in enumerate(p):
            x -= t
            if x <= 0:
                return i
        return len(p) - 1

    def get_ants(self):
        for i in range(self.m):
            self.Table[i][0] = self.start_node
            unvisit = [x for x in range(self.num_city)
                       if x != self.start_node and x != self.end_node]
            current = self.start_node
            j = 1
            while unvisit:
                P = [self.Tau[current][v] ** self.alpha
                     * self.Eta[current][v] ** self.beta for v in unvisit]
                P_sum = sum(P)
                P = [x / P_sum for x in P]
                index = self.rand_choose(P)
                current = unvisit[index]
                self.Table[i][j] = current
                unvisit.remove(current)
                j += 1
            self.Table[i][j] = self.end_node

    def compute_pathlen(self, path):
        result = 0.0
        for i in range(len(path) - 1):
            result += self.dis_mat[path[i]][path[i + 1]]
        return result

    def compute_paths(self, paths):
        return [self.compute_pathlen(one) for one in paths]

    def update_Tau(self):
        delta_tau = np.zeros((self.num_city, self.num_city))
        paths = self.compute_paths(self.Table)
        for i in range(self.m):
            for j in range(self.num_city - 1):
                a = self.Table[i][j]
                b = self.Table[i][j + 1]
                delta_tau[a][b] += self.Q / paths[i]
        self.Tau = (1 - self.rho) * self.Tau + delta_tau

    def aco(self):
        best_len = math.inf
        best_path = None
        for _ in range(self.iter_max):
            self.get_ants()
            self.paths = self.compute_paths(self.Table)
            tmp_len = min(self.paths)
            tmp_path = self.Table[self.paths.index(tmp_len)]
            if tmp_len < best_len:
                best_len = tmp_len
                best_path = tmp_path
            self.update_Tau()
        return best_len, best_path

    def run(self):
        best_length, best_path = self.aco()
        return self.location[best_path], best_length, best_path


def aco(cluster_points, ini_loc, end_loc, data_sizes, cfg,
        m=50, iter_max=200):
    """纯欧氏距离的 ACO 基线。"""
    _seed(cfg)
    full_data, _ = _prepare(cluster_points, ini_loc, end_loc, data_sizes)
    dis_mat = euclidean_dis_mat(full_data)
    n = full_data.shape[0]
    model = _ACO(num_city=n, data=full_data.copy(), dis_mat=dis_mat,
                 m=m, iter_max=iter_max, start_node=0, end_node=n - 1)
    coords, cost, indices = model.run()
    return _order_and_coords(full_data, indices, cost)


def nn_2opt(cluster_points, ini_loc, end_loc, data_sizes, cfg):
    """最近邻构造 + 2-opt 改进（纯欧氏距离）基线。"""
    _seed(cfg)
    full_data, _ = _prepare(cluster_points, ini_loc, end_loc, data_sizes)
    n = full_data.shape[0]

    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_mat[i, j] = float(
                    np.linalg.norm(full_data[i][:2] - full_data[j][:2]))

    # 最近邻构造：从起点 0 出发，终点 n-1 收尾
    route = [0]
    unvisited = set(range(1, n - 1))
    current = 0
    while unvisited:
        nearest = min(unvisited, key=lambda x: dist_mat[current, x])
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    route.append(n - 1)

    # 2-opt 改进
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                old = (dist_mat[route[i - 1], route[i]]
                       + dist_mat[route[j], route[j + 1]])
                new = (dist_mat[route[i - 1], route[j]]
                       + dist_mat[route[i], route[j + 1]])
                if new < old:
                    route[i:j + 1] = route[i:j + 1][::-1]
                    improved = True

    cost = sum(dist_mat[route[i], route[i + 1]]
               for i in range(len(route) - 1))
    return _order_and_coords(full_data, route, cost)
