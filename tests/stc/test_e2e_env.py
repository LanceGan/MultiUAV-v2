"""Unit tests for the shared-bandwidth end-to-end MADRL environment."""

import numpy as np

from segment.stc.e2e import SharedBandwidthE2EEnv
from segment.stc.types import RadioSample, STCScenario, ScenarioLimits


class _FlatField:
    n_sectors = 6

    def sample(self, position):
        return RadioSample(
            a2g_efficiency=np.ones(self.n_sectors),
            g2a_outage=np.zeros(self.n_sectors),
        )


def _scenario(n_tasks=2):
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
    return STCScenario(
        n_uav=2,
        n_tasks=n_tasks,
        task_positions_m=((25.0, 0.0), (150.0, 0.0))[:n_tasks],
        task_data_bits=(1e6, 1e6)[:n_tasks],
        depot_m=(0.0, 0.0),
        terminal_m=(300.0, 0.0),
        limits=limits,
        length_m=300.0,
        width_m=300.0,
        n_bs=2,
        task_seed=42,
        map_seed=42,
    )


def test_observation_shape_and_task_service():
    env = SharedBandwidthE2EEnv(_scenario(), _FlatField())
    obs = env.reset()

    assert len(obs) == 2
    assert obs[0].shape == (env.local_obs_dim,)

    actions = [[50.0, 0.0], [50.0, np.pi]]
    _, _, _, info = env.step(actions)
    assert np.sum(info["visited"]) == 1


def test_same_sector_upload_is_split_equally():
    env = SharedBandwidthE2EEnv(_scenario(), _FlatField())
    env.reset()
    env.buffers[:] = 1e6
    env.visited[:] = True
    env.upload_step[:] = -1

    _, _, _, info = env.step([[0.0, 0.0], [0.0, 0.0]])

    np.testing.assert_allclose(info["uploaded_bits"], [250_000.0, 250_000.0])
    assert info["sectors"].tolist() == [0, 0]


def test_metrics_require_all_uploads_and_no_collision():
    env = SharedBandwidthE2EEnv(_scenario(), _FlatField())
    env.reset()
    env.visited[:] = True
    env.arrival_step[:] = 1
    env.upload_step[:] = 2
    metrics = env.metrics()

    assert metrics["feasible"] is True
    assert metrics["visited_tasks"] == 2
    assert metrics["uploaded_tasks"] == 2
