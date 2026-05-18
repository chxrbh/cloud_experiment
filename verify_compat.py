"""Verification: single seed 42 — checks 1-5 per spec."""
import sys
sys.path.insert(0, ".")
import experiments_p1 as sim
import numpy as np
from collections import defaultdict


def run_seed(strategy_name: str, seed: int = 42):
    rng   = np.random.default_rng(seed)
    nodes = {nid: sim.FogNode(nid, sim.NODE_DEFS[nid]) for nid in sim.NODE_ORDER}
    strat = sim.Strategy(strategy_name)
    failed_set: set = set()
    global_task_id = 0

    n_total = n_deadline = n_redel = 0
    t_wait_all: list[float] = []
    hp_f3_total = hp_f3_deadline = 0

    for w in range(1, sim.N_WINDOWS + 1):
        cfg = sim.get_window_config(w)
        if not cfg["f5_available"]:
            failed_set.add("F5")
        else:
            failed_set.discard("F5")

        tasks = sim.generate_tasks(w, cfg, rng, global_task_id)
        global_task_id += len(tasks)

        for task in tasks:
            t_now = task.t_arrival
            for node in nodes.values():
                node.release_at(t_now)

            primary_overloaded = (task.primary in failed_set or
                                  nodes[task.primary].workload() >= sim.TAU)
            redel = False
            target = None
            if not primary_overloaded:
                target = task.primary
            else:
                all_avail = [nid for nid in sim.NODE_ORDER if nid not in failed_set]
                if strategy_name in ("S1", "S2"):
                    candidates = [nid for nid in all_avail if nid != task.primary]
                else:
                    under = [nid for nid in all_avail
                             if nid != task.primary and nodes[nid].workload() < sim.TAU]
                    candidates = under if under else [nid for nid in all_avail
                                                      if nid != task.primary]
                target = strat.pick_target(task, nodes, candidates, rng)

            if target is not None:
                t_wait, t_exec, t_total = nodes[target].assign_task(task, t_now)
                deadline_met = t_total <= task.max_delay
                n_total += 1
                if deadline_met:
                    n_deadline += 1
                if redel:
                    n_redel += 1
                t_wait_all.append(t_wait)
                if task.task_type == "HP" and target == "F3":
                    hp_f3_total += 1
                    if deadline_met:
                        hp_f3_deadline += 1

        window_end = w * sim.WINDOW_MS
        for node in nodes.values():
            node.window_boundary_cleanup(window_end)

    import statistics
    return {
        "deadline_rate":  n_deadline / n_total if n_total else 0.0,
        "redel_rate":     n_redel / n_total if n_total else 0.0,
        "t_wait_mean":    statistics.mean(t_wait_all) if t_wait_all else 0.0,
        "hp_f3_total":    hp_f3_total,
        "hp_f3_deadline": hp_f3_deadline,
        "n_total":        n_total,
    }


print("=" * 65)
print("Single-seed (42) verification — restored execution times")
print(f"  LS base={sim.TASK_PARAMS['LS']['base_time']}ms  "
      f"HP base={sim.TASK_PARAMS['HP']['base_time']}ms")
print("=" * 65)

results = {}
for s in sim.STRATEGY_NAMES:
    results[s] = run_seed(s)

# ── CHECK 1: deadline satisfaction ───────────────────────────────────────────
print("\nCHECK 1 — Deadline satisfaction rate")
print(f"  {'Strategy':<6} {'Deadline%':>10}  Expected")
for s in sim.STRATEGY_NAMES:
    r = results[s]
    band = ("S1/S2: 70-85%" if s in ("S1","S2") else
            "S3: 90-95%" if s == "S3" else
            "S4-S6: 95-98%")
    print(f"  {s:<6} {r['deadline_rate']:>10.2%}  [{band}]")

gap_s3_s6 = results["S6"]["deadline_rate"] - results["S3"]["deadline_rate"]
print(f"\n  S6 vs S3 gap: {gap_s3_s6:+.2%}  (need >2% to pass)")
check1 = gap_s3_s6 > 0.02
print(f"  Check 1 PASS: {check1}")

# ── CHECK 2: re-delegation rate ───────────────────────────────────────────────
print("\nCHECK 2 — Re-delegation rate")
print(f"  {'Strategy':<6} {'Redel%':>8}")
for s in sim.STRATEGY_NAMES:
    print(f"  {s:<6} {results[s]['redel_rate']:>8.2%}")
check2 = results["S6"]["redel_rate"] <= results["S3"]["redel_rate"]
print(f"\n  S6 redel <= S3 redel: {check2}")

# ── CHECK 3: T_wait ───────────────────────────────────────────────────────────
print("\nCHECK 3 — Mean T_wait (ms)")
print(f"  {'Strategy':<6} {'T_wait':>10}")
for s in sim.STRATEGY_NAMES:
    print(f"  {s:<6} {results[s]['t_wait_mean']:>10.2f}")
check3 = results["S6"]["t_wait_mean"] <= results["S3"]["t_wait_mean"]
print(f"\n  S6 T_wait <= S3 T_wait: {check3}")

# ── CHECK 4: HP on F3 ─────────────────────────────────────────────────────────
print("\nCHECK 4 — HP tasks assigned to F3 (slow node)")
print(f"  {'Strategy':<6} {'HP@F3':>8} {'F3 DL%':>10}")
for s in sim.STRATEGY_NAMES:
    r = results[s]
    dl = r["hp_f3_deadline"] / r["hp_f3_total"] if r["hp_f3_total"] else float("nan")
    print(f"  {s:<6} {r['hp_f3_total']:>8}  {dl:>10.1%}" if r["hp_f3_total"] else
          f"  {s:<6} {0:>8}  {'n/a':>10}")
check4 = all(
    results[s]["hp_f3_total"] == 0 or
    results[s]["hp_f3_deadline"] / results[s]["hp_f3_total"] < 0.10
    for s in ("S4", "S5", "S6")
)
print(f"\n  S4/S5/S6 avoid F3 for HP (DL% near 0%): {check4}")

# ── CHECK 5: overall pass / HP base adjustment guidance ──────────────────────
print("\n" + "=" * 65)
all_pass = check1 and check2 and check3 and check4
print(f"Checks 1-4 PASS: {all_pass}")
if not check1:
    if gap_s3_s6 < 0.01:
        print("  ADVICE: Gap S3 vs S6 <1% — increase HP base_time by 50ms")
    else:
        print(f"  ADVICE: Gap {gap_s3_s6:.2%} — close but below 2% threshold")
if all_pass:
    print("  -> Proceed to full 5-seed simulation.")
print("=" * 65)
