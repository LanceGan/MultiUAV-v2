"""Edge-time-sector conflict attribution (Task 8).

Builds a ``ConflictReport`` from the evaluator's per-slot timelines: total
overlap time, exclusive-vs-shared lost bits, per-UAV loss, and (placeholder)
edge hotspots and pairwise windows.
"""

from segment.stc.types import ConflictReport


def build_conflict_report(result, bandwidth_hz, delta_t_s):
    timelines = result.timelines
    n_uav = len(timelines)

    by_time = {}
    for u, tl in enumerate(timelines):
        for slot in tl.slots:
            by_time.setdefault(slot.time_start_s, []).append((u, slot))

    total_overlap = 0.0
    lost_bits = 0.0
    per_uav_loss = [0.0] * n_uav
    for t in sorted(by_time):
        recs = by_time[t]
        co = {}
        for (u, slot) in recs:
            if slot.active_uploader:
                co[slot.serving_sector] = co.get(slot.serving_sector, 0) + 1
        for cnt in co.values():
            if cnt > 1:
                total_overlap += (cnt - 1) * delta_t_s
        for (u, slot) in recs:
            if slot.active_uploader:
                lost_rate = slot.rate_bps * (1.0 - slot.allocated_bandwidth_hz / bandwidth_hz)
                lost_bits += lost_rate * delta_t_s
                per_uav_loss[u] += lost_rate * delta_t_s

    return ConflictReport(
        total_overlap_s=total_overlap,
        lost_bits=lost_bits,
        per_uav_loss=tuple(per_uav_loss),
        edge_hotspots={},
        pairwise_windows=(),
    )
