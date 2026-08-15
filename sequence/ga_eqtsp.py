"""Layer-2 communication-aware GA_EQTSP sequence planning.

从单机/多机参考实现迁移 GA 框架（greedy_init / ga_cross / ga_mutate / ga_parent /
ga_choose / run），起点固定在 ``full_data`` 下标 0、终点固定在 ``num_city - 1``，
中间下标 1..K 对应 ``cluster_points`` 的 0..K-1。

距离矩阵两种：
- ``euclidean_dis_mat``：纯欧氏距离（无通信）。
- ``comm_dis_mat``：``dis_mat[i][j] = radio_path.edge_cost(full_data[i][:2],
  full_data[j][:2], data_size_i, cfg)["cost"]``，适应度 ``= 1 / path_cost``。

由于 ``edge_cost`` 几何部分（d/E/t/C）沿 i↔j 对称、仅时延惩罚随起点数据量变化，
``comm_dis_mat`` 对每个无序对只采样一次无线电地图（data_size=0 得到基础代价与
容量 C_ij），再分别叠加两端数据量的时延惩罚，结果与逐边调用 ``edge_cost`` 完全一致
（``cost(ds) = base + λτ·max(0, ds/C_ij − τ_max)``），但采样量减半。
"""

import math
import random

import numpy as np

from sequence import radio_path


def _prepare(cluster_points, ini_loc, end_loc, data_sizes):
    """拼接起点 + 巡检点 + 终点，返回 (full_data, data_sizes)。"""
    cluster_points = np.asarray(cluster_points, dtype=float)
    ini_loc = np.asarray(ini_loc, dtype=float).reshape(-1)
    end_loc = np.asarray(end_loc, dtype=float).reshape(-1)
    data_sizes = np.asarray(data_sizes, dtype=float).reshape(-1)
    full_data = np.vstack([ini_loc, cluster_points, end_loc])
    return full_data, data_sizes


def euclidean_dis_mat(full_data):
    """纯欧氏距离矩阵（对角/自身设为 inf）。"""
    n = full_data.shape[0]
    dis_mat = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                dis_mat[i][j] = float(
                    np.linalg.norm(full_data[i][:2] - full_data[j][:2]))
    return dis_mat


def comm_dis_mat(full_data, data_sizes, cfg):
    """通信感知距离矩阵。

    ``dis_mat[i][j]`` 表示从点 i 飞往点 j 并卸载点 i 数据量的综合代价。起点（下标 0）
    与终点（下标 n-1）本身无待卸载数据（data_size=0）。
    """
    n = full_data.shape[0]
    k = len(data_sizes)
    tau_max = float(cfg["tau_max"])
    lam_tau = float(cfg["lambda_tau"])

    dis_mat = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            # 几何/能耗/中断基础代价（data_size=0，时延惩罚为 0）+ 容量 C_ij
            info = radio_path.edge_cost(
                full_data[i][:2], full_data[j][:2], 0.0, cfg)
            base = float(info["cost"])
            c_ij = float(info["C_ij"])

            ds_i = float(data_sizes[i - 1]) if 0 < i < n - 1 else 0.0
            ds_j = float(data_sizes[j - 1]) if 0 < j < n - 1 else 0.0

            dis_mat[i][j] = base + lam_tau * max(0.0, ds_i / c_ij - tau_max)
            dis_mat[j][i] = base + lam_tau * max(0.0, ds_j / c_ij - tau_max)
    return dis_mat


class GA(object):
    """迁移自参考实现的 GA 框架（起点/终点锁定）。"""

    def __init__(self, num_city, num_total, iteration, data, dis_mat):
        self.num_city = num_city
        self.num_total = num_total
        self.iteration = iteration
        self.location = data
        self.dis_mat = dis_mat
        self.ga_choose_ratio = 0.2
        self.mutate_ratio = 0.05
        self.fruits = self.greedy_init(self.dis_mat, num_total, num_city)
        scores = self.compute_adp(self.fruits)
        sort_index = np.argsort(-scores)
        self.iter_x = [0]
        self.iter_y = [1.0 / scores[sort_index[0]]]

    def greedy_init(self, dis_mat, num_total, num_city):
        """贪心初始化，锁定起点 0 和终点 num_city-1，带随机扰动保证多样性。"""
        result = []
        for _ in range(num_total):
            rest = [x for x in range(1, num_city - 1)]
            current = 0
            result_one = [0]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min and np.random.rand() > 0.1:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                if tmp_choose != -1:
                    current = tmp_choose
                    rest.remove(tmp_choose)
                    result_one.append(tmp_choose)
                else:
                    choose = rest.pop()
                    result_one.append(choose)
                    current = choose
            result_one.append(num_city - 1)
            result.append(result_one)
        return result

    def compute_pathlen(self, path):
        result = 0.0
        for i in range(len(path) - 1):
            result += self.dis_mat[path[i]][path[i + 1]]
        return result

    def compute_adp(self, fruits):
        return np.array([1.0 / (self.compute_pathlen(f) + 1e-12)
                         for f in fruits])

    def _fitness(self, path):
        return 1.0 / (self.compute_pathlen(path) + 1e-12)

    def ga_cross(self, x, y):
        """PMX 交叉，只交换中间段、锁定首尾。"""
        len_ = len(x)
        assert len(x) == len(y)
        path_list = list(range(1, len_ - 1))
        if len(path_list) < 2:
            return list(x), list(y)

        order = list(random.sample(path_list, 2))
        order.sort()
        start, end = order

        tmp = x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = y.index(sub)
            if not (start <= index < end):
                x_conflict_index.append(index)

        y_conflict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = x.index(sub)
            if not (start <= index < end):
                y_conflict_index.append(index)

        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp

        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_conflict_index[index]
            y[i], x[j] = x[j], y[i]

        return list(x), list(y)

    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores).copy()
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]
        parents = [self.fruits[i] for i in sort_index]
        parents_score = [scores[i] for i in sort_index]
        return parents, parents_score

    def ga_choose(self, genes_score, genes_choose):
        """轮盘赌选择。"""
        sum_score = sum(genes_score)
        score_ratio = [s * 1.0 / sum_score for s in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        index1 = index2 = 0
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    def ga_mutate(self, gene):
        """逆序变异，锁定首尾。"""
        len_ = len(gene)
        path_list = list(range(1, len_ - 1))
        if len(path_list) < 2:
            return list(gene)
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        tmp = gene[start:end]
        tmp = tmp[::-1]
        gene[start:end] = tmp
        return list(gene)

    def ga(self):
        scores = self.compute_adp(self.fruits)
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]

        fruits = parents.copy()
        stuck_counter = 0
        while len(fruits) < self.num_total:
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)

            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)

            x_adp = self._fitness(gene_x_new)
            y_adp = self._fitness(gene_y_new)

            if stuck_counter < 50:
                if x_adp > y_adp and gene_x_new not in fruits:
                    fruits.append(gene_x_new)
                    stuck_counter = 0
                elif x_adp <= y_adp and gene_y_new not in fruits:
                    fruits.append(gene_y_new)
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                # 组合数耗尽时允许重复个体，避免死循环
                fruits.append(gene_x_new if x_adp > y_adp else gene_y_new)
                stuck_counter = 0

        self.fruits = fruits
        return tmp_best_one, tmp_best_score

    def run(self):
        best_score = -math.inf
        best_list = None
        for _ in range(1, self.iteration + 1):
            tmp_best_one, tmp_best_score = self.ga()
            self.iter_x.append(_)
            self.iter_y.append(1.0 / tmp_best_score)
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                best_list = tmp_best_one
        return self.location[best_list], self.compute_pathlen(best_list), best_list


def _order_and_coords(full_data, best_indices, cost):
    n = full_data.shape[0]
    order_indices = [int(idx) - 1 for idx in best_indices
                     if idx != 0 and idx != n - 1]
    ordered_points = full_data[best_indices]
    return ordered_points, order_indices, float(cost)


def ga_eqtsp(cluster_points, ini_loc, end_loc, data_sizes, cfg,
             num_total=25, iteration=200):
    """通信感知 GA_EQTSP 序列规划。

    Returns:
        (ordered_points, order_indices, cost)：ordered_points 为起点+巡检点+终点的
        有序坐标；order_indices 为巡检点相对 cluster_points 的访问次序（0..K-1）；
        cost 为全程通信感知代价。
    """
    np.random.seed(int(cfg["seed"]))
    random.seed(int(cfg["seed"]))

    full_data, data_sizes = _prepare(cluster_points, ini_loc, end_loc, data_sizes)
    dis_mat = comm_dis_mat(full_data, data_sizes, cfg)
    model = GA(num_city=full_data.shape[0], num_total=num_total,
               iteration=iteration, data=full_data.copy(), dis_mat=dis_mat)
    coords, cost, indices = model.run()
    return _order_and_coords(full_data, indices, cost)
