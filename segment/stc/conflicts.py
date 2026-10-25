"""Edge-time-sector conflict attribution (spec §6).

Builds a :class:`ConflictReport` from the evaluator's per-slot timelines:

- ``total_overlap_s`` — system conflict intensity ``sum (N_m - 1) * dt`` over
  shared sectors;
- ``lost_bits`` — exclusive-vs-shared rate loss ``sum (rate_bps - rate) * dt``
  over active uploaders in shared sectors;
- ``per_uav_loss`` — the same loss partitioned per UAV;
- ``edge_hotspots`` — lost bits keyed by the exact edge occurrence
  ``(uav_id, route_edge_index, source, target, candidate_id)`` so a
  destroy/path-switch operator can target one occurrence;
- ``pairwise_windows`` — merged pairwise conflict intervals (adjacent slots for
  the same pair/sector/route-edges collapse into one window).

Conservation: ``lost_bits`` equals the sum over UAVs and the sum over edge
occurrences within tolerance.
"""

from segment.stc.types import ConflictReport, EdgeOccurrence, PairwiseConflictWindow


def _merge_windows(events):
    """Merge per-slot pair events into maximal pairwise conflict intervals.

    ``events`` is an iterable of ``(uav_a, uav_b, sector, edge_a, edge_b,
    start_s, end_s, lost_bits)``.  Events sharing the same pair/sector/edges
    whose intervals are time-adjacent collapse into one window.
    """
    groups = {}
    for (a, b, sec, ea, eb, start, end, lost) in events:
        groups.setdefault((a, b, sec, ea, eb), []).append((start, end, lost))
    windows = []
    for (a, b, sec, ea, eb), segs in sorted(groups.items()):
        segs.sort()
        cur_start, cur_end, cur_lost = segs[0]
        for start, end, lost in segs[1:]:
            if abs(start - cur_end) < 1e-9:
                cur_end = end
                cur_lost += lost
            else:
                windows.append(PairwiseConflictWindow(
                    a, b, sec, ea, eb, cur_start, cur_end, cur_lost))
                cur_start, cur_end, cur_lost = start, end, lost
        windows.append(PairwiseConflictWindow(
            a, b, sec, ea, eb, cur_start, cur_end, cur_lost))
    windows.sort(key=lambda w: (w.uav_a, w.uav_b, w.sector, w.start_s))
    return tuple(windows)


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
    edge_loss = {}
    events = []

    for t in sorted(by_time):
        recs = by_time[t]
        co = {}
        for (u, slot) in recs:
            if slot.active_uploader:
                co[slot.serving_sector] = co.get(slot.serving_sector, 0) + 1

        for cnt in co.values():
            if cnt > 1:
                total_overlap += (cnt - 1) * delta_t_s

        # lost bits only for active uploaders in a shared sector (a lone
        # uploader has allocated == full bandwidth, so its loss is zero).
        for (u, slot) in recs:
            if slot.active_uploader and co.get(slot.serving_sector, 0) > 1:
                lost_rate = slot.rate_bps * (1.0 - slot.allocated_bandwidth_hz / bandwidth_hz)
                lost = lost_rate * delta_t_s
                lost_bits += lost
                per_uav_loss[u] += lost
                occ = EdgeOccurrence(uav_id=u, route_edge_index=slot.edge_id,
                                     source=slot.source, target=slot.target,
                                     candidate_id=slot.candidate_id)
                edge_loss[occ] = edge_loss.get(occ, 0.0) + lost

        # pairwise events for this slot
        by_sector = {}
        for (u, slot) in recs:
            if slot.active_uploader:
                by_sector.setdefault(slot.serving_sector, []).append((u, slot))
        for sec, members in by_sector.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    ua, sa = members[i]
                    ub, sb = members[j]
                    la = sa.rate_bps * (1.0 - sa.allocated_bandwidth_hz / bandwidth_hz) * delta_t_s
                    lb = sb.rate_bps * (1.0 - sb.allocated_bandwidth_hz / bandwidth_hz) * delta_t_s
                    if ua < ub:
                        a, b, ea, eb = ua, ub, sa.edge_id, sb.edge_id
                    else:
                        a, b, ea, eb = ub, ua, sb.edge_id, sa.edge_id
                    events.append((a, b, sec, ea, eb,
                                   sa.time_start_s, sa.time_end_s, la + lb))

    return ConflictReport(
        total_overlap_s=total_overlap,
        lost_bits=lost_bits,
        per_uav_loss=tuple(per_uav_loss),
        edge_hotspots=edge_loss,
        pairwise_windows=_merge_windows(events),
    )
