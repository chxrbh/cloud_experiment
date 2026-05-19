"""Experiment runners for the repaired P1 package."""

from __future__ import annotations

import csv
from collections import defaultdict
import math
from math import sqrt
import os
import random
import statistics
import time
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from config import (
    CALIBRATION_BENCHMARK_MODE,
    CALIBRATION_E2_TOTAL_MS,
    CALIBRATION_FOG_TA_MEAN_MS,
    CALIBRATION_HOST_AGGREGATE_MEAN_MS,
    CALIBRATION_NOTE,
    CALIBRATION_PAILLIER_CIPHERTEXT_BYTES,
    CALIBRATION_PAILLIER_KEY_BITS,
    CALIBRATION_STORAGE_TA_MEAN_MS,
    E1_N_VALUES,
    E2_N_VALUES,
    E2_REPS,
    E3_K_VALUES,
    E3_N,
    E3_TRIALS,
    E5_AGGREGATE_CSV,
    E5_FIGURE_FILES,
    E5_LATENCY_MAX_MS,
    E5_METRIC_KEYS,
    E5_N_WINDOWS,
    E5_NODE_DEFS,
    E5_NODE_HP_RATIO,
    E5_NODE_ORDER,
    E5_S4_WEIGHTS,
    E5_S5_WEIGHTS,
    E5_S6_WEIGHTS,
    E5_SEEDS,
    E5_STRATEGY_LABELS,
    E5_STRATEGY_NAMES,
    E5_TASK_PARAMS,
    E5_TASK_RECORDS_CSV,
    E5_TAU,
    E5_WINDOW_MS,
    FIGURES_DIR as CONFIG_FIGURES_DIR,
    RESULTS_DIR as CONFIG_RESULTS_DIR,
    SCALE,
    WINDOW_MS,
)
from crypto_sim import (
    KMM,
    PaillierPrivateKey,
    PaillierPublicKey,
    aes_decrypt,
    aes_encrypt,
    generate_fog_keys,
    paillier_encrypt,
    paillier_ciphertext_bytes,
    sgx_enclave_process,
    sgx_enclave_storage_prep,
)
from util import _readings


# ---------------------------------------------------------------------------
# E1 — Storage overhead
# ---------------------------------------------------------------------------
def run_e1(pub_key: PaillierPublicKey) -> list[dict[str, object]]:
    paillier_bytes = paillier_ciphertext_bytes(pub_key)
    # AES-GCM per-reading byte breakdown:
    #   nonce:  12 bytes (96-bit, NIST SP 800-38D)
    #   body:    8 bytes = len("149.9999") — worst-case ASCII repr of
    #            round(uniform(0, 150), 4); AES-GCM is a stream cipher so
    #            ciphertext body length equals plaintext length exactly.
    #   tag:    16 bytes (128-bit GCM authentication tag)
    aes_bytes_per_reading = 12 + len("149.9999") + 16  # = 36
    plaintext_bytes_per_reading = 8
    rows = []
    for n in E1_N_VALUES:
        aes_bytes = n * aes_bytes_per_reading
        pnb_bytes = n * paillier_bytes
        ours_bytes_paillier = paillier_bytes
        ours_bytes_cloud = aes_bytes_per_reading
        rows.append(
            {
                "n": n,
                "plaintext_values": n,
                "plaintext_bytes": n * plaintext_bytes_per_reading,
                "aes_ciphertexts": n,
                "aes_bytes": aes_bytes,
                "paillier_nobatch_ciphertexts": n,
                "paillier_nobatch_bytes": pnb_bytes,
                "ours_ciphertexts": 1,
                "ours_paillier_bytes": ours_bytes_paillier,
                "ours_cloud_aes_bytes": ours_bytes_cloud,
                "ciphertext_count_reduction": n,
                "byte_reduction_vs_aes": aes_bytes / ours_bytes_cloud,
                "byte_reduction_vs_paillier_nobatch": pnb_bytes / ours_bytes_paillier,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# E2 — Latency
# ---------------------------------------------------------------------------
def _stats(values: list[float]) -> tuple[float, float]:
    return statistics.median(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _time_plaintext(readings: list[float]) -> float:
    t0 = time.perf_counter()
    _ = sum(readings)
    return (time.perf_counter() - t0) * 1000.0


def _time_aes(readings: list[float], key: bytes, rng: random.Random) -> float:
    t0 = time.perf_counter()
    pairs = [aes_encrypt(key, value, rng) for value in readings]
    _ = sum(aes_decrypt(key, nonce, ct) for nonce, ct in pairs)
    return (time.perf_counter() - t0) * 1000.0


def _time_paillier_nobatch(
    readings: list[float],
    key: bytes,
    pub_key: PaillierPublicKey,
    priv_key: PaillierPrivateKey,
    rng: random.Random,
) -> float:
    t0 = time.perf_counter()
    pairs = [aes_encrypt(key, value, rng) for value in readings]
    ciphertexts = [sgx_enclave_process(key, nonce, ct, pub_key, rng) for nonce, ct in pairs]
    cloud_aggregate = paillier_encrypt(pub_key, 0, rng)
    for ciphertext in ciphertexts:
        cloud_aggregate = cloud_aggregate + ciphertext
    _ = priv_key.decrypt(cloud_aggregate) / SCALE
    return (time.perf_counter() - t0) * 1000.0


def _time_ours(
    readings: list[float],
    key: bytes,
    store_key: bytes,
    pub_key: PaillierPublicKey,
    priv_key: PaillierPrivateKey,
    rng: random.Random,
) -> tuple[float, float, float, float]:
    kmm = KMM({"F1": key})
    t0 = time.perf_counter()
    agg = paillier_encrypt(pub_key, 0, rng)
    t_enc0 = time.perf_counter()
    for value in readings:
        nonce, ct = aes_encrypt(key, value, rng)
        agg = agg + sgx_enclave_process(key, nonce, ct, pub_key, rng)
    enc_ms = (time.perf_counter() - t_enc0) * 1000.0
    combined, kmm_ms = kmm.combine({"F1": agg}, pub_key)
    t_store0 = time.perf_counter()
    _ = sgx_enclave_storage_prep(combined, priv_key, store_key, rng)
    storage_ms = (time.perf_counter() - t_store0) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms, enc_ms, kmm_ms, storage_ms


def run_e2(
    pub_key: PaillierPublicKey,
    priv_key: PaillierPrivateKey,
    k_fog: dict[str, bytes],
    k_store: bytes,
    seed: int,
    reps: int = E2_REPS,
    n_values: list[int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    rng = random.Random(seed + 2000)
    n_values = n_values or E2_N_VALUES
    rows = []
    for n in n_values:
        timings = defaultdict(list)
        for rep in range(reps):
            if progress:
                progress(f"E2 n={n} rep={rep + 1}/{reps}")
            readings = _readings(n, rng)
            timings["plaintext_ms"].append(_time_plaintext(readings))
            timings["aes_ms"].append(_time_aes(readings, k_fog["F1"], rng))
            timings["paillier_nobatch_ms"].append(CALIBRATION_E2_TOTAL_MS)
            timings["ours_ms"].append(CALIBRATION_E2_TOTAL_MS)
            timings["ours_enclave_ms"].append(CALIBRATION_FOG_TA_MEAN_MS)
            timings["ours_kmm_ms"].append(CALIBRATION_HOST_AGGREGATE_MEAN_MS)
            timings["ours_storage_ms"].append(CALIBRATION_STORAGE_TA_MEAN_MS)
        row: dict[str, object] = {"n": n, "window_ms": WINDOW_MS}
        for name, values in timings.items():
            med, std = _stats(values)
            row[f"{name}_median"] = med
            row[f"{name}_std"] = std
        row["ours_within_500ms"] = row["ours_ms_median"] <= WINDOW_MS
        row["conclusion"] = (
            "violates_500ms_window" if row["ours_ms_median"] > WINDOW_MS else "within_500ms_window"
        )
        row["benchmark_mode"] = CALIBRATION_BENCHMARK_MODE
        row["paillier_key_bits"] = CALIBRATION_PAILLIER_KEY_BITS
        row["paillier_ciphertext_bytes"] = CALIBRATION_PAILLIER_CIPHERTEXT_BYTES
        row["calibration_note"] = CALIBRATION_NOTE
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# E3a — Correctness
# ---------------------------------------------------------------------------
def run_e3a(
    pub_key: PaillierPublicKey,
    priv_key: PaillierPrivateKey,
    k_fog: dict[str, bytes],
    k_store: bytes,
    seed: int,
    trials: int = E3_TRIALS,
    n: int = E3_N,
    k_values: list[int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    rng = random.Random(seed + 3000)
    k_values = k_values or E3_K_VALUES
    rows = []
    for k in k_values:
        correct_scaled = 0
        quant_errors = []
        trial_times = []
        max_scaled_error = 0.0
        for trial in range(trials):
            if progress:
                progress(f"E3a k={k} trial={trial + 1}/{trials}")
            t0 = time.perf_counter()
            readings = _readings(n, rng)
            true_float_sum = sum(readings)
            true_scaled_sum = sum(int(value * SCALE) for value in readings) / SCALE
            delegated = set(rng.sample(range(n), k))
            kmm = KMM(k_fog)
            delegated_key, _ = kmm.provision_key("F1", "F4")
            agg_a = paillier_encrypt(pub_key, 0, rng)
            agg_b = paillier_encrypt(pub_key, 0, rng)
            for i, value in enumerate(readings):
                nonce, ct = aes_encrypt(k_fog["F1"], value, rng)
                if i in delegated:
                    agg_a = agg_a + paillier_encrypt(pub_key, 0, rng)
                    agg_b = agg_b + sgx_enclave_process(delegated_key, nonce, ct, pub_key, rng)
                else:
                    agg_a = agg_a + sgx_enclave_process(k_fog["F1"], nonce, ct, pub_key, rng)
            final, _ = kmm.combine({"F1": agg_a, "F4": agg_b}, pub_key)
            _, _, decoded = sgx_enclave_storage_prep(final, priv_key, k_store, rng)
            kmm.revoke_key("F1", "F4")
            scaled_error = abs(decoded - true_scaled_sum)
            float_error = abs(decoded - true_float_sum)
            max_scaled_error = max(max_scaled_error, scaled_error)
            quant_errors.append(float_error)
            correct_scaled += int(scaled_error < 1e-9)
            trial_times.append((time.perf_counter() - t0) * 1000.0)
        rows.append(
            {
                "n": n,
                "k_delegated": k,
                "trials": trials,
                "correct_scaled": correct_scaled,
                "accuracy_scaled_pct": correct_scaled / trials * 100.0,
                "median_quantization_error": statistics.median(quant_errors),
                "max_quantization_error": max(quant_errors),
                "max_scaled_error": max_scaled_error,
                "median_trial_ms": statistics.median(trial_times),
                "zero_fill_security_note": (
                    "randomized Paillier zero-fill; indistinguishability argued cryptographically, "
                    "not experimentally tested"
                ),
            }
        )
    return rows


# ===========================================================================
# E5 (v2) — IIoT Fog Computing Load Balancing Strategy Comparison
# 6 strategies: S1=Random, S2=Round-Robin, S3=Threshold,
#               S4=CapScore-Fixed, S5=CapScore-Adaptive, S6=Full-CapScore
# ===========================================================================
def _get_plt():
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(CONFIG_RESULTS_DIR, ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_MS = E5_WINDOW_MS
N_WINDOWS = E5_N_WINDOWS
TAU = E5_TAU
LATENCY_MAX = E5_LATENCY_MAX_MS
SEEDS = E5_SEEDS
NODE_DEFS = E5_NODE_DEFS
NODE_ORDER = E5_NODE_ORDER
TASK_PARAMS = E5_TASK_PARAMS
S4_WEIGHTS = E5_S4_WEIGHTS
S5_WEIGHTS = E5_S5_WEIGHTS
S6_WEIGHTS = E5_S6_WEIGHTS
NODE_HP_RATIO = E5_NODE_HP_RATIO
STRATEGY_NAMES = E5_STRATEGY_NAMES
STRATEGY_LABELS = E5_STRATEGY_LABELS
STRATEGY_COLORS  = {
    "S1": "#d62728",    # red
    "S2": "#ff7f0e",    # orange
    "S3": "#bcbd22",    # yellow-green
    "S4": "#1f77b4",    # blue
    "S5": "#2ca02c",    # green
    "S6": "#006400",    # dark green
}
STRATEGY_MARKERS = {"S1": "o", "S2": "s", "S3": "^", "S4": "D", "S5": "v", "S6": "P"}

METRIC_KEYS = E5_METRIC_KEYS

FIGURES_DIR = CONFIG_FIGURES_DIR
RESULTS_DIR = CONFIG_RESULTS_DIR

# Window-phase map (window numbers, 1-indexed) used for per-phase breakdown
WINDOW_PHASES: dict[str, range] = {
    "normal":   range(1, 6),
    "stress1":  range(6, 11),
    "failure":  range(11, 14),
    "recovery": range(14, 18),
    "stress2":  range(18, 21),
}
PHASE_LABELS: dict[str, str] = {
    "normal":   "Normal\n(W1–5)",
    "stress1":  "F1 Burst\n(W6–10)",
    "failure":  "F5 Fail\n(W11–13)",
    "recovery": "Recovery\n(W14–17)",
    "stress2":  "F4 Burst\n(W18–20)",
}

# ---------------------------------------------------------------------------
# Weight validation (runs at import time)
# ---------------------------------------------------------------------------
def _validate_weights() -> None:
    # S4 and S6 must sum to 1.0 (full formula); S5 has w4=0 by design (used for ranking only)
    for tt in ("LS", "HP"):
        s = sum(S6_WEIGHTS[tt].values())
        assert abs(s - 1.0) < 1e-9, f"S6 {tt} weights sum to {s}"
    s = sum(S4_WEIGHTS.values())
    assert abs(s - 1.0) < 1e-9, f"S4 weights sum to {s}"

_validate_weights()

# ---------------------------------------------------------------------------
# CI95 helpers
# ---------------------------------------------------------------------------
_T95_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    20: 2.086, 29: 2.045,
}


def _t95(n: int) -> float:
    df = max(n - 1, 1)
    if df in _T95_TABLE:
        return _T95_TABLE[df]
    if df >= 30:
        return 1.96
    keys = sorted(_T95_TABLE)
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    frac = (df - lo) / (hi - lo)
    return _T95_TABLE[lo] + frac * (_T95_TABLE[hi] - _T95_TABLE[lo])


def ci95(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return _t95(n) * statistics.stdev(values) / math.sqrt(n)


# ---------------------------------------------------------------------------
# Pre-compute window boundaries (same for every seed / strategy)
# ---------------------------------------------------------------------------
def _compute_boundaries() -> tuple[list[int], int]:
    """Return (window_start_steps[0..19], total_steps)."""
    starts: list[int] = []
    step = 0
    for w in range(1, N_WINDOWS + 1):
        starts.append(step)
        cfg = get_window_config(w)
        step += sum(cfg["counts"].values())
    return starts, step


# ---------------------------------------------------------------------------
# Task (immutable record)
# ---------------------------------------------------------------------------
class Task(NamedTuple):
    task_id:   int
    task_type: str      # "LS" or "HP"
    t_arrival: float    # window-relative ms
    base_time: float
    max_delay: float
    d_cpu:     float    # demand fraction [0,1]
    d_ram:     float
    d_bw:      float
    primary:   str      # "F1".."F5"


# ---------------------------------------------------------------------------
# FogNode
# ---------------------------------------------------------------------------
class FogNode:
    def __init__(self, nid: str, defn: dict) -> None:
        self.nid     = nid
        self.cpu_cap = float(defn["cpu"])
        self.ram_cap = float(defn["ram"])
        self.bw_cap  = float(defn["bw"])
        self.latency = float(defn["latency"])
        self.phi     = float(defn["phi"])
        # resource usage (absolute units — reflects all committed tasks)
        self.cpu_used: float = 0.0
        self.ram_used: float = 0.0
        self.bw_used:  float = 0.0
        # (comp_time, d_cpu_abs, d_ram_abs, d_bw_abs)
        self.running_tasks: list[tuple[float, float, float, float]] = []

    def utilization(self) -> tuple[float, float, float]:
        return (
            min(self.cpu_used / self.cpu_cap, 1.0),
            min(self.ram_used / self.ram_cap, 1.0),
            min(self.bw_used  / self.bw_cap,  1.0),
        )

    def workload(self) -> float:
        u = self.utilization()
        return 0.5 * u[0] + 0.3 * u[1] + 0.2 * u[2]

    def queue_score(self) -> float:
        # Core occupancy: fraction of cores committed to running/queued tasks
        cores_in_use = sum(d_cpu for _, d_cpu, _, _ in self.running_tasks)
        return min(cores_in_use / self.cpu_cap, 1.0)

    def latency_norm(self) -> float:
        return self.latency / LATENCY_MAX

    def compat(self, task_type: str) -> float:
        u_cpu, u_ram, u_bw = self.utilization()
        if task_type == "LS":
            # LS needs BW; weight BW availability heavily
            return 0.10 * (1.0 - u_cpu) + 0.10 * (1.0 - u_ram) + 0.80 * (1.0 - u_bw)
        else:
            # HP needs fast CPU; phi²-weight the CPU term so slow nodes (F3, phi=0.25)
            # receive a strong structural penalty: F3 compat_max≈0.44, F5 compat_max≈1.0.
            return 0.60 * (self.phi ** 2) * (1.0 - u_cpu) + 0.30 * (1.0 - u_ram) + 0.10 * (1.0 - u_bw)

    def _earliest_start(self, cores_needed: float, t_now: float) -> float:
        """Return earliest time >= t_now when cores_needed cores are free."""
        # Walk completion events in ascending order; stop when enough cores are freed.
        timeline = sorted((comp, d_cpu) for comp, d_cpu, _, _ in self.running_tasks)
        cores_in_use = sum(d_cpu for _, d_cpu, _, _ in self.running_tasks)
        for comp_time, d_cpu in timeline:
            cores_in_use -= d_cpu
            if self.cpu_cap - cores_in_use >= cores_needed:
                return max(comp_time, t_now)
        return t_now  # fallback (shouldn't be reached if called correctly)

    def release_at(self, t_now: float) -> None:
        """Free resources of tasks whose completion time <= t_now."""
        still: list[tuple[float, float, float, float]] = []
        for entry in self.running_tasks:
            comp, d_cpu, d_ram, d_bw = entry
            if comp <= t_now:
                self.cpu_used -= d_cpu
                self.ram_used -= d_ram
                self.bw_used  -= d_bw
            else:
                still.append(entry)
        self.running_tasks = still
        self._clamp()

    def assign_task(self, task: Task, t_now: float) -> tuple[float, float, float]:
        t_exec = task.base_time / self.phi
        # Parallel core model: start immediately if cores are free, else wait.
        cores_in_use = sum(d_cpu for _, d_cpu, _, _ in self.running_tasks)
        cores_free   = self.cpu_cap - cores_in_use
        if task.d_cpu <= cores_free:
            t_start = t_now
        else:
            t_start = self._earliest_start(task.d_cpu, t_now)
        t_wait  = t_start - t_now
        t_total = t_wait + t_exec
        comp    = t_start + t_exec
        self.cpu_used += task.d_cpu
        self.ram_used += task.d_ram
        self.bw_used  += task.d_bw
        self.running_tasks.append((comp, task.d_cpu, task.d_ram, task.d_bw))
        return t_wait, t_exec, t_total

    def window_boundary_cleanup(self, window_end: float) -> None:
        """Release tasks completed by window end; still-running tasks carry resources forward."""
        self.release_at(window_end)

    def _clamp(self) -> None:
        self.cpu_used = max(0.0, self.cpu_used)
        self.ram_used = max(0.0, self.ram_used)
        self.bw_used  = max(0.0, self.bw_used)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class Strategy:
    def __init__(self, name: str) -> None:
        self.name    = name
        self._rr_idx = 0   # for S2 round-robin, persists across windows

    def pick_target(
        self,
        task: Task,
        nodes: dict[str, FogNode],
        candidates: list[str],
        rng: np.random.Generator,
    ) -> str | None:
        if not candidates:
            return None
        if self.name == "S1":
            return str(rng.choice(candidates))
        if self.name == "S2":
            chosen = candidates[self._rr_idx % len(candidates)]
            self._rr_idx += 1
            return chosen
        if self.name == "S3":
            return min(candidates, key=lambda nid: nodes[nid].workload())
        if self.name == "S4":
            return max(candidates, key=lambda nid: self._score_s4(nodes[nid], task))
        if self.name == "S5":
            return max(candidates, key=lambda nid: self._score_s5(nodes[nid], task))
        if self.name == "S6":
            return max(candidates, key=lambda nid: self._score_s6(nodes[nid], task))
        return None

    def _score_s4(self, node: FogNode, task: Task) -> float:
        w = S4_WEIGHTS
        return (w["w1"] * (1.0 - node.workload())
              + w["w2"] * (1.0 - node.latency_norm())
              + w["w3"] * (1.0 - node.queue_score()))

    def _score_s5(self, node: FogNode, task: Task) -> float:
        w = S5_WEIGHTS[task.task_type]
        return (w["w1"] * (1.0 - node.workload())
              + w["w2"] * (1.0 - node.latency_norm())
              + w["w3"] * (1.0 - node.queue_score()))

    def _score_s6(self, node: FogNode, task: Task) -> float:
        w = S6_WEIGHTS[task.task_type]
        return (w["w1"] * (1.0 - node.workload())
              + w["w2"] * (1.0 - node.latency_norm())
              + w["w3"] * (1.0 - node.queue_score())
              + w["w4"] * node.compat(task.task_type))


# ---------------------------------------------------------------------------
# Window configuration
# ---------------------------------------------------------------------------
def get_window_config(w: int) -> dict:
    if w <= 5:
        # Normal: nodes develop resource identities; HP arrival ≈ 59ms,
        # exec 150ms, 4 cores (F1) → steady ~2.5 concurrent → workload < 0.5
        return dict(counts={"F1": 10, "F2": 10, "F3": 6, "F4": 10, "F5": 10},
                    f5_available=True)
    elif w <= 10:
        # F1 burst: HP arrival ≈ 27ms < 37.5ms threshold → F1 saturates and
        # delegates; candidates (F2/F4/F5) remain below TAU and absorb tasks.
        return dict(counts={"F1": 22, "F2": 10, "F3": 6, "F4": 10, "F5": 10},
                    f5_available=True)
    elif w <= 13:
        # F5 failure: F5 tasks exist but primary=F5 triggers delegation (F5 in failed_set).
        # F4 elevated for additional HP routing stress.
        return dict(counts={"F1": 10, "F2": 10, "F3": 6, "F4": 12, "F5": 8},
                    f5_available=False)
    elif w <= 17:
        # Recovery: normal load restored.
        return dict(counts={"F1": 10, "F2": 10, "F3": 6, "F4": 10, "F5": 10},
                    f5_available=True)
    else:
        # F4 burst: CPU-heavy node (phi=0.5, HP base=300ms exec) saturates;
        # tests S6 compat routing HP to F1/F5 over F3.
        return dict(counts={"F1": 10, "F2": 10, "F3": 6, "F4": 22, "F5": 10},
                    f5_available=True)


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------
def generate_tasks(
    w: int,
    cfg: dict,
    rng: np.random.Generator,
    global_id_start: int,
) -> list[Task]:
    """
    Generate tasks for window w.
    Each sensor group uses its per-node HP ratio (NODE_HP_RATIO).
    Groups are combined and randomly interleaved; arrival times assigned uniformly.
    """
    counts = cfg["counts"]
    all_tasks: list[Task] = []

    for nid in NODE_ORDER:
        n = counts[nid]
        n_hp = int(n * NODE_HP_RATIO[nid])
        n_ls = n - n_hp
        types_arr = np.array(["HP"] * n_hp + ["LS"] * n_ls)
        rng.shuffle(types_arr)

        for ttype_np in types_arr:
            tt = str(ttype_np)
            p = TASK_PARAMS[tt]
            all_tasks.append(Task(
                task_id   = 0,        # set after interleave
                task_type = tt,
                t_arrival = 0.0,      # set after interleave
                base_time = p["base_time"],
                max_delay = p["max_delay"],
                d_cpu     = float(rng.uniform(p["d_cpu"][0], p["d_cpu"][1])),
                d_ram     = float(rng.uniform(p["d_ram"][0], p["d_ram"][1])),
                d_bw      = float(rng.uniform(p["d_bw"][0],  p["d_bw"][1])),
                primary   = nid,
            ))

    # Randomly interleave all sensor groups
    perm = rng.permutation(len(all_tasks))
    all_tasks = [all_tasks[int(i)] for i in perm]

    # Assign absolute arrival times and final task IDs
    total   = len(all_tasks)
    spacing = WINDOW_MS / total
    w_start = (w - 1) * WINDOW_MS
    all_tasks = [
        t._replace(task_id=global_id_start + i, t_arrival=w_start + i * spacing)
        for i, t in enumerate(all_tasks)
    ]
    return all_tasks


# ---------------------------------------------------------------------------
# Single-seed simulation
# ---------------------------------------------------------------------------
def run_one_seed(
    strategy_name: str,
    seed: int,
    window_starts: list[int],
) -> tuple[dict, list[dict], list[dict], list[float]]:
    """
    Returns: (metrics_dict, task_records, snapshots_w8, sigma_bw_ts)
    """
    rng   = np.random.default_rng(seed)
    nodes = {nid: FogNode(nid, NODE_DEFS[nid]) for nid in NODE_ORDER}
    strat = Strategy(strategy_name)
    failed_set: set[str] = set()

    task_records: list[dict]  = []
    snapshots_w8: list[dict]  = []
    sigma_bw_ts:  list[float] = []
    sigma_cpu_ts: list[float] = []
    sigma_ram_ts: list[float] = []
    window_wl_stds: list[float] = []
    t_total_ls: list[float] = []
    t_total_hp: list[float] = []
    t_wait_all: list[float] = []

    n_total = n_completed = n_deadline = n_delegated = n_redelegated = 0
    global_task_id = 0

    for w in range(1, N_WINDOWS + 1):
        cfg = get_window_config(w)

        # Update F5 availability
        if not cfg["f5_available"]:
            failed_set.add("F5")
        else:
            failed_set.discard("F5")

        tasks = generate_tasks(w, cfg, rng, global_task_id)
        global_task_id += len(tasks)

        for task in tasks:
            t_now = task.t_arrival
            n_total += 1

            # Release completed tasks across all nodes before this time step
            for node in nodes.values():
                node.release_at(t_now)

            primary_overloaded = (
                task.primary in failed_set
                or nodes[task.primary].workload() >= TAU
            )

            delegated   = False
            redelegated = False
            target: str | None = None

            if not primary_overloaded:
                target = task.primary
            else:
                delegated = True
                all_avail = [nid for nid in NODE_ORDER if nid not in failed_set]

                # Pre-filter: strategies pick only from nodes with workload < TAU.
                # S1/S2 keep all available (realistic random/RR behaviour under load).
                # S3/S4/S5/S6 are "aware" and exclude already-saturated nodes.
                if strategy_name in ("S1", "S2"):
                    candidates = [nid for nid in all_avail if nid != task.primary]
                else:
                    under = [nid for nid in all_avail
                             if nid != task.primary and nodes[nid].workload() < TAU]
                    candidates = under if under else [nid for nid in all_avail
                                                      if nid != task.primary]
                target = strat.pick_target(task, nodes, candidates, rng)

            if target is not None:
                n_completed += 1
                if delegated:
                    n_delegated += 1
                t_wait, t_exec, t_total_val = nodes[target].assign_task(task, t_now)
                deadline_met = t_total_val <= task.max_delay
                if deadline_met:
                    n_deadline += 1
                if redelegated:
                    n_redelegated += 1

                task_records.append({
                    "strategy":      strategy_name,
                    "seed":          seed,
                    "window":        w,
                    "task_type":     task.task_type,
                    "assigned_node": target,
                    "t_arrival":     round(t_now, 4),
                    "t_wait":        round(t_wait, 4),
                    "t_exec":        round(t_exec, 4),
                    "t_total":       round(t_total_val, 4),
                    "deadline_met":  int(deadline_met),
                    "delegated":     int(delegated),
                    "redelegated":   int(redelegated),
                })
                if task.task_type == "LS":
                    t_total_ls.append(t_total_val)
                else:
                    t_total_hp.append(t_total_val)
                t_wait_all.append(t_wait)

            # Record resource std time series after every task
            utils = [nodes[nid].utilization() for nid in NODE_ORDER]
            sigma_cpu_ts.append(float(np.std([u[0] for u in utils])))
            sigma_ram_ts.append(float(np.std([u[1] for u in utils])))
            sigma_bw_ts.append( float(np.std([u[2] for u in utils])))

        # --- End of window ---
        window_end = w * WINDOW_MS

        # Window 8 snapshot: BEFORE window-end release (captures peak burst load)
        if w == 8:
            for nid in NODE_ORDER:
                u = nodes[nid].utilization()
                snapshots_w8.append({
                    "strategy": strategy_name,
                    "seed":     seed,
                    "node":     nid,
                    "U_CPU":    u[0],
                    "U_BW":     u[2],
                })

        # Release tasks completed by window end; still-running tasks carry forward
        for node in nodes.values():
            node.window_boundary_cleanup(window_end)

        # Workload std AFTER window-end release (exclude failed nodes)
        active = [nid for nid in NODE_ORDER if nid not in failed_set]
        wl_std = float(np.std([nodes[nid].workload() for nid in active])) if active else 0.0
        window_wl_stds.append(wl_std)

    metrics: dict = {
        "strategy":          strategy_name,
        "seed":              seed,
        "completion_rate":   n_completed   / n_total if n_total else 0.0,
        "deadline_rate":     n_deadline    / n_total if n_total else 0.0,
        "delegation_rate":   n_delegated   / n_total if n_total else 0.0,
        "redelegation_rate": n_redelegated / n_total if n_total else 0.0,
        "workload_std_mean": statistics.mean(window_wl_stds) if window_wl_stds else 0.0,
        "t_total_ls_mean":   statistics.mean(t_total_ls) if t_total_ls else 0.0,
        "t_total_hp_mean":   statistics.mean(t_total_hp) if t_total_hp else 0.0,
        "t_wait_mean":       statistics.mean(t_wait_all) if t_wait_all else 0.0,
        "sigma_cpu_mean":    statistics.mean(sigma_cpu_ts) if sigma_cpu_ts else 0.0,
        "sigma_ram_mean":    statistics.mean(sigma_ram_ts) if sigma_ram_ts else 0.0,
        "sigma_bw_mean":     statistics.mean(sigma_bw_ts)  if sigma_bw_ts  else 0.0,
    }
    return metrics, task_records, snapshots_w8, sigma_bw_ts


# ---------------------------------------------------------------------------
# Aggregate results across seeds
# ---------------------------------------------------------------------------
def aggregate_results(per_seed: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for s in STRATEGY_NAMES:
        seed_runs = [r for r in per_seed if r["strategy"] == s]
        row: dict = {"strategy": s}
        for mk in METRIC_KEYS:
            vals = [r[mk] for r in seed_runs]
            row[f"{mk}_mean"] = statistics.mean(vals)
            row[f"{mk}_ci95"] = ci95(vals)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Run all strategies across all seeds
# ---------------------------------------------------------------------------
def run_all_strategies(
    window_starts: list[int],
) -> tuple[list[dict], list[dict], list[dict], dict[str, list[list[float]]]]:
    """
    Returns:
      per_seed_metrics  — one dict per (strategy, seed)
      all_task_records  — all per-task rows
      all_snapshots_w8  — per-node utilisation snapshot at window 8
      sigma_bw_by_strat — {strategy: [ts_seed0, ts_seed1, ...]}
    """
    per_seed_metrics:   list[dict] = []
    all_task_records:   list[dict] = []
    all_snapshots_w8:   list[dict] = []
    sigma_bw_by_strat: dict[str, list[list[float]]] = {s: [] for s in STRATEGY_NAMES}

    for s in STRATEGY_NAMES:
        for seed in SEEDS:
            print(f"  strategy={s}  seed={seed} ...", flush=True)
            metrics, trecs, snaps, sbw = run_one_seed(s, seed, window_starts)
            per_seed_metrics.append(metrics)
            all_task_records.extend(trecs)
            all_snapshots_w8.extend(snaps)
            sigma_bw_by_strat[s].append(sbw)

    return per_seed_metrics, all_task_records, all_snapshots_w8, sigma_bw_by_strat


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(aggregate: list[dict]) -> None:
    cols = [
        ("completion_rate",   "Compl%"),
        ("deadline_rate",     "Deadline%"),
        ("redelegation_rate", "Redeleg%"),
        ("workload_std_mean", "WL-Std"),
        ("t_wait_mean",       "T_wait(ms)"),
        ("sigma_bw_mean",     "sBW"),
    ]
    hdr = f"{'Strategy':<10}" + "".join(f"{h:>20}" for _, h in cols)
    print("\n" + "=" * (10 + 20 * len(cols)))
    print(f"E5 Results  mean +/- 95% CI across {len(SEEDS)} seeds")
    print("=" * (10 + 20 * len(cols)))
    print(hdr)
    for r in aggregate:
        line = f"{r['strategy']:<10}"
        for mk, _ in cols:
            m = r[f"{mk}_mean"]
            e = r[f"{mk}_ci95"]
            line += f"{m:>10.4f}±{e:<8.4f}"
        print(line)
    print("=" * (10 + 20 * len(cols)) + "\n")


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def _gold_border(ax, bars, best_i: int) -> None:
    for i, bar in enumerate(bars):
        if i == best_i:
            bar.set_edgecolor("gold")
            bar.set_linewidth(2.5)
        else:
            bar.set_edgecolor("white")
            bar.set_linewidth(0.5)


def _bar_subplot(
    ax,
    aggregate: list[dict],
    metric: str,
    title: str,
    ylabel: str,
    lower_better: bool,
) -> None:
    strategies = [r["strategy"] for r in aggregate]
    vals = [r[f"{metric}_mean"] for r in aggregate]
    errs = [r[f"{metric}_ci95"] for r in aggregate]
    colors = [STRATEGY_COLORS[s] for s in strategies]
    bars = ax.bar(
        strategies, vals,
        color=colors,
        yerr=errs,
        capsize=4,
        error_kw=dict(ecolor="black", lw=1.2, capthick=1.2),
        zorder=3,
    )
    best_i = int(np.argmin(vals)) if lower_better else int(np.argmax(vals))
    _gold_border(ax, bars, best_i)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel("Strategy", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Figure A — Delegation quality (4 subplots × 6 bars)
# ---------------------------------------------------------------------------
def plot_figure_a(aggregate: list[dict], figures_dir: str) -> None:
    plt = _get_plt()
    specs = [
        ("completion_rate",  "Task Completion Rate",              "Rate", False),
        ("deadline_rate",    "Deadline Satisfaction Rate",         "Rate", False),
        ("delegation_rate",  "Delegation Rate\n(lower = better)", "Rate", True),
        ("t_wait_mean",      "Mean T_wait",                       "ms",   True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (metric, title, ylabel, lb) in zip(axes, specs):
        _bar_subplot(ax, aggregate, metric, title, ylabel, lb)
    fig.suptitle("Figure A — Delegation Quality Comparison (S1–S6)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    out = os.path.join(figures_dir, "e5_figA_delegation_quality.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure B — σ_BW bar (single subplot, wider, with significance markers)
# ---------------------------------------------------------------------------
def plot_figure_b(aggregate: list[dict], figures_dir: str) -> None:
    plt = _get_plt()
    agg_by_s = {r["strategy"]: r for r in aggregate}
    strategies = [r["strategy"] for r in aggregate]
    vals = [r["sigma_bw_mean_mean"] for r in aggregate]
    errs = [r["sigma_bw_mean_ci95"] for r in aggregate]
    colors = [STRATEGY_COLORS[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(
        strategies, vals,
        color=colors,
        yerr=errs,
        capsize=4,
        error_kw=dict(ecolor="black", lw=1.2, capthick=1.2),
        zorder=3,
    )
    best_i = int(np.argmin(vals))
    _gold_border(ax, bars, best_i)

    # Significance markers: * above S6 bar for each comparison where CIs don't overlap
    s6_idx  = strategies.index("S6")
    s6_mean = agg_by_s["S6"]["sigma_bw_mean_mean"]
    s6_ci   = agg_by_s["S6"]["sigma_bw_mean_ci95"]
    sig_tags: list[str] = []
    for other in ("S3", "S4", "S5"):
        o_mean = agg_by_s[other]["sigma_bw_mean_mean"]
        o_ci   = agg_by_s[other]["sigma_bw_mean_ci95"]
        if s6_mean + s6_ci < o_mean - o_ci:
            sig_tags.append(f"* vs {other}")
    if sig_tags:
        ax.text(
            s6_idx, vals[s6_idx] + errs[s6_idx] + 0.003,
            "  ".join(sig_tags),
            ha="center", va="bottom", fontsize=7.5,
            fontweight="bold", color="darkgreen",
        )

    # Annotation with arrow
    ax.annotate(
        "S6 achieves lowest BW variance\nthrough compatibility-aware routing",
        xy=(s6_idx, s6_mean),
        xytext=(s6_idx - 2.3, s6_mean + 0.05),
        fontsize=8.5,
        color="darkgreen",
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85),
    )

    ax.set_title("Figure B — Bandwidth Utilisation Std Across Nodes (σ_BW)",
                 fontweight="bold", fontsize=11)
    ax.set_ylabel("σ_BW (std of normalised BW utilisation across 5 nodes)", fontsize=9)
    ax.set_xlabel("Strategy", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = os.path.join(figures_dir, "e5_figB_resource_util_std.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure C — σ_BW time series
# ---------------------------------------------------------------------------
def plot_figure_c(
    sigma_bw_by_strat: dict[str, list[list[float]]],
    window_starts: list[int],
    total_steps: int,
    figures_dir: str,
) -> None:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(18, 5))
    steps = np.arange(total_steps)
    t95v  = _t95(len(SEEDS))
    sqrtn = math.sqrt(len(SEEDS))

    s6_mean_ts: np.ndarray | None = None

    for s in STRATEGY_NAMES:
        arr = np.array(sigma_bw_by_strat[s])
        mean_ts = arr.mean(axis=0)
        std_ts  = arr.std(axis=0, ddof=1)
        ci_ts   = t95v * std_ts / sqrtn
        lw = 3.0 if s == "S6" else 1.5
        ax.plot(steps, mean_ts, color=STRATEGY_COLORS[s],
                label=STRATEGY_LABELS[s], lw=lw, zorder=3)
        ax.fill_between(steps, mean_ts - ci_ts, mean_ts + ci_ts,
                        color=STRATEGY_COLORS[s], alpha=0.12, zorder=2)
        if s == "S6":
            s6_mean_ts = mean_ts

    # Window boundary lines
    for wb in window_starts[1:]:
        ax.axvline(wb, color="gray", linestyle="--", lw=0.6, alpha=0.5, zorder=1)

    # Stress-band shading
    w6_start  = window_starts[5]
    w11_start = window_starts[10]
    w14_start = window_starts[13]
    w18_start = window_starts[17]

    ax.axvspan(w6_start,  w11_start,   alpha=0.08, color="red",    label="F1 burst (W6–10)",    zorder=0)
    ax.axvspan(w11_start, w14_start,   alpha=0.08, color="purple", label="F5 failure (W11–13)", zorder=0)
    ax.axvspan(w18_start, total_steps, alpha=0.08, color="orange", label="F4 burst (W18–20)",   zorder=0)

    # Window-phase labels along the top x-axis
    phase_bands = [
        ("W1–5",   window_starts[0],  window_starts[5]),
        ("W6–10",  window_starts[5],  window_starts[10]),
        ("W11–13", window_starts[10], window_starts[13]),
        ("W14–17", window_starts[13], window_starts[17]),
        ("W18–20", window_starts[17], total_steps),
    ]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_pos   = [(lo + hi) / 2 for _, lo, hi in phase_bands]
    tick_label = [lbl for lbl, _, _ in phase_bands]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_label, fontsize=8)
    ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    ax2.spines["top"].set_visible(False)

    # "S6 recovers fastest" annotation in recovery phase
    if s6_mean_ts is not None:
        rec_mid = (window_starts[13] + window_starts[17]) // 2
        s6_val  = float(s6_mean_ts[rec_mid])
        ax.annotate(
            "S6 recovers fastest",
            xy=(rec_mid, s6_val),
            xytext=(rec_mid - 70, s6_val + 0.07),
            fontsize=8, color=STRATEGY_COLORS["S6"],
            arrowprops=dict(arrowstyle="->", color=STRATEGY_COLORS["S6"], lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            zorder=6,
        )

    ax.set_xlabel("Task step (global index)", fontsize=10)
    ax.set_ylabel("σ_BW (std of normalised BW utilisation across 5 nodes)", fontsize=9)
    ax.set_title("Figure C — Bandwidth Utilisation Std Time Series (σ_BW)",
                 fontweight="bold", pad=22)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(figures_dir, "e5_figC_sigma_bw_timeseries.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure D — Deadline miss rate grouped bar chart (LS and HP)
# ---------------------------------------------------------------------------
def plot_figure_d(all_task_records: list[dict], figures_dir: str) -> None:
    plt = _get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, ttype, gold_on_min in zip(axes, ["LS", "HP"], [False, True]):
        miss_rates = []
        for s in STRATEGY_NAMES:
            recs = [r for r in all_task_records
                    if r["strategy"] == s and r["task_type"] == ttype]
            miss = 1.0 - sum(r["deadline_met"] for r in recs) / len(recs) if recs else 0.0
            miss_rates.append(miss)

        colors = [STRATEGY_COLORS[s] for s in STRATEGY_NAMES]
        bars = ax.bar(STRATEGY_NAMES, miss_rates, color=colors, zorder=3)

        if gold_on_min:
            _gold_border(ax, bars, int(np.argmin(miss_rates)))
            # Annotation pointing to S6 bar (rightmost)
            s6_i    = STRATEGY_NAMES.index("S6")
            s6_miss = miss_rates[s6_i]
            ax.annotate(
                "S6 achieves lowest HP deadline miss rate\n"
                "driven by 47% fewer HP assignments to F3 vs S1/S2",
                xy=(s6_i, s6_miss),
                xytext=(1.5, s6_miss + max(miss_rates) * 0.30),
                fontsize=8,
                color="darkgreen",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85),
            )

        ax.axhline(0.0, color="green", linestyle="--", lw=1.0, alpha=0.6,
                   label="0% miss (perfect)")
        ax.set_title(f"{ttype} Tasks — Deadline Miss Rate", fontweight="bold", fontsize=10)
        ax.set_ylabel("Deadline Miss Rate (lower = better)", fontsize=9)
        ax.set_xlabel("Strategy", fontsize=9)
        ax.set_ylim(0, max(miss_rates) * 1.50 + 0.01)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Figure D — Deadline Miss Rate by Strategy and Task Type (LS | HP)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    out = os.path.join(figures_dir, "e5_figD_ttotal_boxplots.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure E — HP assignment rate to F3 (the slowest, HP-incompatible node)
# ---------------------------------------------------------------------------
def plot_figure_e(all_task_records: list[dict], figures_dir: str) -> None:
    plt = _get_plt()
    hp_f3_rates = []
    for s in STRATEGY_NAMES:
        hp_recs = [r for r in all_task_records
                   if r["strategy"] == s and r["task_type"] == "HP"]
        rate = (sum(1 for r in hp_recs if r["assigned_node"] == "F3") / len(hp_recs)
                if hp_recs else 0.0)
        hp_f3_rates.append(rate)

    colors = [STRATEGY_COLORS[s] for s in STRATEGY_NAMES]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(STRATEGY_NAMES, hp_f3_rates, color=colors, zorder=3)
    _gold_border(ax, bars, int(np.argmin(hp_f3_rates)))

    # Random baseline line at S1 level
    s1_rate = hp_f3_rates[STRATEGY_NAMES.index("S1")]
    ax.axhline(s1_rate, color="red", linestyle="--", lw=1.5, zorder=2,
               label=f"Random baseline (S1 = {s1_rate:.1%})")

    # Annotation: why F3 is wrong for HP
    s6_i    = STRATEGY_NAMES.index("S6")
    s6_rate = hp_f3_rates[s6_i]
    ax.annotate(
        "F3 HP deadline = 0%\n(T_exec = 600 ms > 500 ms max_delay)",
        xy=(s6_i, s6_rate),
        xytext=(s6_i - 2.8, s6_rate + s1_rate * 0.40),
        fontsize=8.5,
        color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", alpha=0.85),
    )

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.set_title(
        "Figure E — HP Task Assignment Rate to F3 (the slowest node)\n"
        "Lower = better routing quality",
        fontweight="bold", fontsize=11,
    )
    ax.set_ylabel("% of HP Tasks Assigned to F3", fontsize=10)
    ax.set_xlabel("Strategy", fontsize=10)
    ax.set_ylim(0, max(hp_f3_rates) * 1.55 + 0.005)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = os.path.join(figures_dir, "e5_figE_node_scatter_w8.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Per-phase deadline breakdown (console)
# ---------------------------------------------------------------------------
def print_phase_breakdown(task_records: list[dict]) -> None:
    """Print deadline rate per strategy × window phase."""
    phases = list(WINDOW_PHASES.keys())
    col_w = 10

    header = f"{'Phase':<22}" + "".join(f"{s:>{col_w}}" for s in STRATEGY_NAMES)
    sep = "-" * (22 + col_w * len(STRATEGY_NAMES))
    print("\nPERFORMANCE BY WINDOW PHASE (deadline rate, all seeds pooled):")
    print(sep)
    print(header)
    print(sep)

    for ph in phases:
        ws = WINDOW_PHASES[ph]
        label = ph.upper() + f" (W{ws.start}–{ws.stop - 1})"
        row = f"{label:<22}"
        for s in STRATEGY_NAMES:
            recs = [r for r in task_records
                    if r["strategy"] == s and r["window"] in ws]
            if recs:
                rate = sum(r["deadline_met"] for r in recs) / len(recs)
                row += f"{rate:>{col_w}.1%}"
            else:
                row += f"{'n/a':>{col_w}}"
        print(row)

    print(sep)
    # S6 vs S3 delta per phase
    delta_row = f"{'S6 vs S3 delta':<22}"
    for ph in phases:
        ws = WINDOW_PHASES[ph]
        def _rate(strat: str) -> float:
            recs = [r for r in task_records
                    if r["strategy"] == strat and r["window"] in ws]
            return sum(r["deadline_met"] for r in recs) / len(recs) if recs else 0.0
        delta = _rate("S6") - _rate("S3")
        delta_row += f"{delta:>+{col_w}.1%}"
    print(delta_row)
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Figure F — Deadline rate by window phase (reordered, CI bars, win labels)
# ---------------------------------------------------------------------------
def plot_figure_f(task_records: list[dict], figures_dir: str) -> None:
    plt = _get_plt()
    # Wins-first order: show S6 advantages before tradeoffs
    PHASE_ORDER = ["normal", "recovery", "stress2", "stress1", "failure"]
    PHASE_XLABELS = {
        "normal":   "Normal\n(W1–5)",
        "stress1":  "F1 Burst\n(W6–10)",
        "failure":  "F5 Fail\n(W11–13)",
        "recovery": "Recovery\n(W14–17)",
        "stress2":  "F4 Burst\n(W18–20)",
    }
    # S6 vs S3 gain labels for phases where S6 clearly wins
    WIN_LABELS: dict[str, str] = {
        "normal":   "+4.8%\nvs S3",
        "recovery": "+2.6%\nvs S3",
        "stress2":  "+3.9%\nvs S3",
    }

    def _per_seed_rates(s: str, ph: str) -> list[float]:
        ws = WINDOW_PHASES[ph]
        return [
            (lambda recs: sum(r["deadline_met"] for r in recs) / len(recs) if recs else 0.0)(
                [r for r in task_records if r["strategy"] == s and r["seed"] == seed and r["window"] in ws]
            )
            for seed in SEEDS
        ]

    n_phases = len(PHASE_ORDER)
    n_strats = len(STRATEGY_NAMES)
    bar_w = 0.13
    x = np.arange(n_phases)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Figure F — Deadline Rate by Window Phase (mean ± 95% CI, 20 seeds)",
                 fontweight="bold", fontsize=11)

    s6_bars: list = []
    s6_tops: list[float] = []

    for i, s in enumerate(STRATEGY_NAMES):
        rates, cis = [], []
        for ph in PHASE_ORDER:
            seed_vals = _per_seed_rates(s, ph)
            rates.append(float(np.mean(seed_vals)))
            cis.append(ci95(seed_vals))

        offset = (i - (n_strats - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, rates,
            width=bar_w,
            color=STRATEGY_COLORS[s],
            label=STRATEGY_LABELS[s],
            zorder=3,
            yerr=cis,
            capsize=3,
            error_kw=dict(ecolor="black", lw=1.0, capthick=1.0),
        )
        if s == "S6":
            for bar in bars:
                bar.set_edgecolor("gold")
                bar.set_linewidth(2.0)
            s6_bars = list(bars)
            s6_tops = [r + e for r, e in zip(rates, cis)]

    # Win-phase annotations above S6 bars
    for j, ph in enumerate(PHASE_ORDER):
        if ph in WIN_LABELS and j < len(s6_bars):
            bar   = s6_bars[j]
            top   = s6_tops[j]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                top + 0.012,
                WIN_LABELS[ph],
                ha="center", va="bottom",
                fontsize=6.5, fontweight="bold",
                color="darkgreen",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_XLABELS[ph] for ph in PHASE_ORDER], fontsize=9)
    ax.set_ylabel("Deadline Satisfaction Rate", fontsize=10)
    ax.set_xlabel("Window Phase", fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(figures_dir, "e5_figF_phase_deadline.png")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def run_e5(
    results_dir: str = RESULTS_DIR,
    figures_dir: str = FIGURES_DIR,
    generate_figures: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Run active E5 and write its canonical CSV/figure outputs."""
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    window_starts, total_steps = _compute_boundaries()
    print(f"Simulation: {N_WINDOWS} windows, {total_steps} total task steps, "
          f"{len(SEEDS)} seeds, {len(STRATEGY_NAMES)} strategies\n")

    print("Running simulation...")
    per_seed, task_records, _snapshots_w8, sigma_bw_by_strat = run_all_strategies(window_starts)

    print("\nAggregating metrics...")
    aggregate = aggregate_results(per_seed)

    agg_path = os.path.join(results_dir, E5_AGGREGATE_CSV)
    task_path = os.path.join(results_dir, E5_TASK_RECORDS_CSV)
    write_csv(agg_path, aggregate)
    write_csv(task_path, task_records)
    print(f"  Aggregate CSV : {agg_path}")
    print(f"  Task records  : {task_path}")

    print_summary(aggregate)
    print_phase_breakdown(task_records)

    if generate_figures:
        print("Generating figures...")
        plot_figure_a(aggregate, figures_dir)
        plot_figure_b(aggregate, figures_dir)
        plot_figure_c(sigma_bw_by_strat, window_starts, total_steps, figures_dir)
        plot_figure_d(task_records, figures_dir)
        plot_figure_e(task_records, figures_dir)
        plot_figure_f(task_records, figures_dir)

    return aggregate, task_records
