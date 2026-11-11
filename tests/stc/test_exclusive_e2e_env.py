"""Acceptance tests for the exclusive-sector fixed-route environment."""

import numpy as np

from segment.stc.e2e_exclusive import ExclusiveSectorFixedRouteEnv
from segment.stc.types import RadioSample, STCScenario, ScenarioLimits


class _FlatField:
    n_sectors = 6

    def sample(self, position):
        return RadioSample(
            a2g_efficiency=np.ones(self.n_sectors),
            g2a_outage=np.zeros(self.n_sectors),
        )


class _PersistentOutageField(_FlatField):
    def sample(self, position):
        return RadioSample(
            a2g_efficiency=np.ones(self.n_sectors),
            g2a_outage=np.full(self.n_sectors, 0.9),
        )


class _PositionOutageField(_FlatField):
    def sample(self, position):
        outage = 0.9 if float(position[0]) < 25.0 else 0.1
        return RadioSample(
            a2g_efficiency=np.ones(self.n_sectors),
            g2a_outage=np.full(self.n_sectors, outage),
        )


def _scenario(n_uav=2, task_positions=((50.0, 0.0), (200.0, 0.0))):
    limits = ScenarioLimits(
        delta_t_s=0.5,
        v_max_mps=50.0,
        uav_height_m=100.0,
        arrive_radius_m=15.0,
        max_wait_s=60.0,
        mission_deadline_s=7200.0,
        bandwidth_hz=1e6,
        uplink_power_w=0.2,
        eps_out=0.35,
        w_t=0.5,
        w_d=0.5,
    )
    n_tasks = len(task_positions)
    return STCScenario(
        n_uav=n_uav,
        n_tasks=n_tasks,
        task_positions_m=tuple(task_positions),
        task_data_bits=tuple(1e6 for _ in range(n_tasks)),
        depot_m=(0.0, 0.0),
        terminal_m=(300.0, 0.0),
        limits=limits,
        length_m=300.0,
        width_m=300.0,
        n_bs=2,
        task_seed=42,
        map_seed=42,
    )


def test_sector_is_exclusive_and_bandwidth_is_not_split():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(), _FlatField(), {0: [0], 1: [1]}
    )
    env.buffers[:] = 1e6
    env._service_and_terminal = lambda: np.zeros(env.n_uav)

    _, _, _, info = env.step([[0.0, 0.0], [0.0, 0.0]])

    assert len(set(info["sectors"].tolist())) == env.n_uav
    np.testing.assert_allclose(info["uploaded_bits"], [500_000.0, 500_000.0])
    assert info["sector_conflicts"] == 0


def test_fixed_route_controls_visit_order():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(), _FlatField(), {0: [1, 0], 1: []}
    )

    assert env._target_kind(0) == ("task", 1)
    assert env.routes[0] == (1, 0)


def test_task_requires_entry_within_ten_metres():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(n_uav=1, task_positions=((10.5, 0.0),)),
        _FlatField(),
        {0: [0]},
    )
    env.positions[0] = np.asarray([0.0, 0.0])
    env.step([[0.0, 0.0]])
    assert not bool(env.visited[0])

    env.positions[0] = np.asarray([0.6, 0.0])
    _, _, _, info = env.step([[0.0, 0.0]])
    assert bool(info["visited"][0])


def test_upload_must_finish_before_entering_next_service_region():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(n_uav=1, task_positions=((10.0, 0.0), (50.0, 0.0))),
        _FlatField(),
        {0: [0, 1]},
    )
    env.positions[0] = np.asarray([0.1, 0.0])
    env.step([[0.0, 0.0]])
    assert bool(env.visited[0])
    assert env.buffers[0] > 0.0

    env.positions[0] = np.asarray([40.1, 0.0])
    _, _, _, info = env.step([[0.0, 0.0]])
    assert not bool(info["visited"][1])
    assert info["early_service_arrivals"] == 1


def test_final_upload_is_required_before_terminal_completion():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(n_uav=1, task_positions=((10.0, 0.0),)),
        _FlatField(),
        {0: [0]},
    )
    env.positions[0] = np.asarray([0.1, 0.0])
    env.step([[0.0, 0.0]])
    assert env.buffers[0] > 0.0

    env.positions[0] = np.asarray([290.1, 0.0])
    _, _, _, info = env.step([[0.0, 0.0]])
    assert not bool(info["done"][0])
    assert info["early_terminal_arrivals"] == 1
    assert "terminal_not_reached" in env.metrics()["failure_reasons"]


def test_progress_reward_favours_motion_towards_guided_target():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(n_uav=1, task_positions=((100.0, 0.0),)),
        _FlatField(),
        {0: [0]},
    )
    env.step([[50.0, 0.0]])
    towards = env.reward_term_last["progress"][0]

    env.reset()
    env.positions[0] = np.asarray([50.0, 50.0])
    env.step([[50.0, np.pi]])
    away = env.reward_term_last["progress"][0]

    assert towards > 0.0
    assert away < 0.0


def test_mapguide_targets_safe_upload_ring_before_next_task():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(n_uav=1, task_positions=((10.0, 0.0), (100.0, 0.0))),
        _FlatField(),
        {0: [0, 1]},
    )
    env.positions[0] = np.asarray([0.1, 0.0])
    env.step([[0.0, 0.0]])
    assert bool(env.visited[0])
    assert env.buffers[0] > 0.0

    guide = env.guide_target(0)
    distance = float(np.linalg.norm(guide - env.task_positions[1]))
    assert env.arrival_radius < distance <= env.arrival_radius + 5.0 + 1e-9


def test_early_arrival_and_invalid_stall_have_explicit_negative_terms():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(n_uav=1, task_positions=((10.0, 0.0), (50.0, 0.0))),
        _FlatField(),
        {0: [0, 1]},
    )
    env.positions[0] = np.asarray([0.1, 0.0])
    env.step([[0.0, 0.0]])
    env.positions[0] = np.asarray([40.1, 0.0])
    env.step([[0.0, 0.0]])
    assert env.reward_term_last["early_arrival"][0] < 0.0

    env.reset()
    env.positions[0] = np.asarray([150.0, 150.0])
    for _ in range(22):
        env.step([[0.0, 0.0]])
    assert env.reward_term_totals["invalid_stall"][0] < 0.0
    assert env.max_stall_steps[0] > 20


def test_upload_guard_prevents_entering_next_service_disk_with_buffer():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(n_uav=1, task_positions=((10.0, 0.0), (50.0, 0.0))),
        _FlatField(),
        {0: [0, 1]},
    )
    env.positions[0] = np.asarray([0.1, 0.0])
    env.step([[0.0, 0.0]])
    assert env.buffers[0] > 0.0

    env.positions[0] = np.asarray([35.0, 0.0])
    guide_delta = env.guide_target(0) - env.positions[0]
    guide_bearing = np.arctan2(guide_delta[1], guide_delta[0])
    guarded = env.project_safe_actions(
        [[50.0, -float(guide_bearing)]], 0.2
    )
    np.testing.assert_allclose(guarded[0], env.guide_action(0))
    assert env.last_upload_guard_projections == 1

    env.last_upload_guard_projections = 0
    env.project_safe_actions(
        [[50.0, -float(guide_bearing)]], 0.2, update_stats=False
    )
    assert env.last_upload_guard_projections == 0


def test_feasibility_uses_maximum_contiguous_outage_not_average_ratio():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(n_uav=1, task_positions=((200.0, 0.0),)),
        _PersistentOutageField(),
        {0: [0]},
        outage_max_s=6.0,
    )
    for _ in range(13):
        env.step([[0.0, 0.0]])

    metrics = env.metrics()
    assert metrics["max_contiguous_outage_s"][0] == 6.5
    assert "contiguous_outage_excess" in metrics["failure_reasons"]


def test_contiguous_outage_budget_is_enforced_independently_per_uav():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(n_uav=2, task_positions=((200.0, 0.0), (250.0, 0.0))),
        _PositionOutageField(),
        {0: [0], 1: [1]},
        outage_max_s=20.0,
    )
    for _ in range(41):
        env.step([[0.0, 0.0], [0.0, 0.0]])

    metrics = env.metrics()
    assert metrics["max_contiguous_outage_s"][0] == 20.5
    assert metrics["max_contiguous_outage_s"][1] == 0.0
    assert "contiguous_outage_excess" in metrics["failure_reasons"]


def test_local_observation_exposes_own_qos_timer_and_last_outage():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(n_uav=1, task_positions=((200.0, 0.0),)),
        _PersistentOutageField(),
        {0: [0]},
        outage_max_s=20.0,
    )
    obs = env.reset()
    for _ in range(4):
        obs, _, _, _ = env.step([[0.0, 0.0]])

    assert obs[0].shape == (env.local_obs_dim,)
    np.testing.assert_allclose(obs[0][13], 2.0 / 20.0)
    assert obs[0][14] == 0.9


def test_qos_hard_constraint_penalty_is_applied_once_per_uav():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(n_uav=1, task_positions=((200.0, 0.0),)),
        _PersistentOutageField(),
        {0: [0]},
        outage_max_s=1.0,
    )
    for _ in range(3):
        env.step([[0.0, 0.0]])
    assert env.reward_term_totals["qos"][0] == -10.0
    assert env.reward_term_totals["qos_risk"][0] == 0.0

    for _ in range(3):
        env.step([[0.0, 0.0]])
    assert env.reward_term_totals["qos"][0] == -10.0


def test_safety_projection_replaces_action_without_reliable_sector():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(n_uav=1, task_positions=((100.0, 0.0),)),
        _PersistentOutageField(),
        {0: [0]},
    )
    projected = env.project_safe_actions([[50.0, 0.5]], outage_threshold=0.2)

    assert projected[0, 1] == 0.0
    assert env.last_safety_projections == 1


def test_safety_projection_keeps_action_with_reliable_sector():
    env = ExclusiveSectorFixedRouteEnv(
        _scenario(n_uav=1, task_positions=((100.0, 0.0),)),
        _FlatField(),
        {0: [0]},
    )
    original = np.asarray([[50.0, 0.5]], dtype=float)
    projected = env.project_safe_actions(original, outage_threshold=0.2)

    np.testing.assert_allclose(projected, original)
    assert env.last_safety_projections == 0
