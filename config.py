"""Shared configuration for the repaired P1 IIoT experiments."""

from __future__ import annotations

DEFAULT_SEED = 42
DEFAULT_KEY_BITS = 2048
WINDOW_MS = 500.0
SCALE = 1000
TAU = 0.8
KMM_PROV_MS = 50.0
T_ACK_MS = 100.0
TLS_MS = 20.0
ATTEST_MS = 50.0

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
SEEDS = [42, 43, 44, 45, 46]

E1_N_VALUES = [10, 50, 100, 200, 500, 1000]

# E2 stops at n=500; the proposed method uses a fixed calibrated total
# (444.128 ms) that is within the 500 ms window for all tested n values.
E2_N_VALUES = [10, 50, 100, 200, 500]
E2_REPS = 3

# 2048-bit calibration benchmark:
# secure_iiot_full_crypto_bench 100 2048, host_reference_2048_no_ta.
# OP-TEE QEMU timing only; not physical hardware timing. TA-side 2048-bit
# Paillier is not enabled in this QEMU prototype.
CALIBRATION_BENCHMARK_MODE = "host_reference_2048_no_ta"
CALIBRATION_PAILLIER_KEY_BITS = 2048
CALIBRATION_PAILLIER_CIPHERTEXT_BYTES = 512
CALIBRATION_ITERATIONS = 100
CALIBRATION_FOG_TA_MEAN_US = 227_537
CALIBRATION_HOST_AGGREGATE_MEAN_US = 1_474
CALIBRATION_STORAGE_TA_MEAN_US = 215_117
CALIBRATION_FOG_TA_MEAN_MS = CALIBRATION_FOG_TA_MEAN_US / 1000.0
CALIBRATION_HOST_AGGREGATE_MEAN_MS = CALIBRATION_HOST_AGGREGATE_MEAN_US / 1000.0
CALIBRATION_STORAGE_TA_MEAN_MS = CALIBRATION_STORAGE_TA_MEAN_US / 1000.0
CALIBRATION_E2_TOTAL_MS = (
    CALIBRATION_FOG_TA_MEAN_MS
    + CALIBRATION_HOST_AGGREGATE_MEAN_MS
    + CALIBRATION_STORAGE_TA_MEAN_MS
)
CALIBRATION_NOTE = (
    "2048-bit host reference benchmark; OP-TEE QEMU timing only, not physical "
    "hardware timing. TA-side 2048-bit Paillier is not enabled in this QEMU prototype."
)

E3_N = 100
E3_TRIALS = 100
E3_K_VALUES = [1, 5, 10, 20, 50]

# Legacy four-strategy E5 constants. Retained only for archived result
# reproducibility; active Exp5 uses the E5 constants below.
E5_WINDOWS = 20
E5_TASKS_PER_WINDOW = 100
E5_TO_RATIO = 0.70

# E5: six-strategy task-aware load-balancing simulation.
E5_WINDOW_MS = WINDOW_MS
E5_N_WINDOWS = 20
E5_TAU = TAU
E5_LATENCY_MAX_MS = 40.0
E5_SEEDS = list(range(42, 62))
E5_NODE_ORDER = ["F1", "F2", "F3", "F4", "F5"]
E5_NODE_DEFS: dict[str, dict[str, float]] = {
    "F1": {"cpu": 4, "ram": 8, "bw": 100, "latency": 10.0, "phi": 1.00},
    "F2": {"cpu": 2, "ram": 4, "bw": 50, "latency": 20.0, "phi": 0.50},
    "F3": {"cpu": 1, "ram": 2, "bw": 20, "latency": 40.0, "phi": 0.25},
    "F4": {"cpu": 2, "ram": 4, "bw": 80, "latency": 15.0, "phi": 0.50},
    "F5": {"cpu": 4, "ram": 8, "bw": 100, "latency": 10.0, "phi": 1.00},
}
E5_TASK_PARAMS: dict[str, dict[str, object]] = {
    "LS": {
        "max_delay": 100.0,
        "base_time": 20.0,
        "d_cpu": (0.10, 0.30),
        "d_ram": (0.10, 0.30),
        "d_bw": (5.0, 15.0),
    },
    "HP": {
        "max_delay": 500.0,
        "base_time": 150.0,
        "d_cpu": (0.80, 1.20),
        "d_ram": (0.50, 1.50),
        "d_bw": (1.0, 5.0),
    },
}
E5_S4_WEIGHTS = {"w1": 0.50, "w2": 0.15, "w3": 0.35, "w4": 0.0}
E5_S5_WEIGHTS = {
    "LS": {"w1": 0.15, "w2": 0.45, "w3": 0.15, "w4": 0.0},
    "HP": {"w1": 0.40, "w2": 0.10, "w3": 0.30, "w4": 0.0},
}
E5_S6_WEIGHTS = {
    "LS": {"w1": 0.10, "w2": 0.35, "w3": 0.10, "w4": 0.45},
    "HP": {"w1": 0.15, "w2": 0.30, "w3": 0.10, "w4": 0.45},
}
E5_NODE_HP_RATIO = {
    "F1": 0.85,
    "F2": 0.15,
    "F3": 0.00,
    "F4": 0.85,
    "F5": 0.15,
}
E5_STRATEGY_NAMES = ["S1", "S2", "S3", "S4", "S5", "S6"]
E5_STRATEGY_LABELS = {
    "S1": "S1: Random",
    "S2": "S2: Round-Robin",
    "S3": "S3: Threshold",
    "S4": "S4: CapScore-Fixed",
    "S5": "S5: CapScore-Adaptive",
    "S6": "S6: Proposed",
}
E5_METRIC_KEYS = [
    "completion_rate",
    "deadline_rate",
    "delegation_rate",
    "redelegation_rate",
    "workload_std_mean",
    "t_total_ls_mean",
    "t_total_hp_mean",
    "t_wait_mean",
    "sigma_cpu_mean",
    "sigma_ram_mean",
    "sigma_bw_mean",
]
E5_AGGREGATE_CSV = "e5_results.csv"
E5_TASK_RECORDS_CSV = "e5_task_records.csv"
E5_FIGURE_FILES = [
    "e5_figA_delegation_quality.png",
    "e5_figB_resource_util_std.png",
    "e5_figC_sigma_bw_timeseries.png",
    "e5_figD_ttotal_boxplots.png",
    "e5_figE_node_scatter_w8.png",
    "e5_figF_phase_deadline.png",
]

E3B_SOURCE_N = 30
E3B_SOURCES = ["F1", "F2"]
E3B_BACKUP = "F4"
E3B_BACKUP_OWN_N = 30
E3B_K_DELEGATED_PER_SOURCE = 5
E3B_TRIALS = 100

E4_K_VALUES = [1, 2, 5, 10, 20, 50, 100]
E4_REPS = 30

E6_FAILURE_SCENARIOS = {
    "fail_0ms": 0.0,
    "mid_window_250ms": 250.0,
    "late_window_450ms": 450.0,
}
E6_METHODS = [
    "b1_gossip",
    "b2_replication",
    "checkpoint",
    "b4_multilayer",
    "b5_fog_clustering",
    "proposed_ack_kmm",
]
E6_SEEDS = list(range(30))
E6_CHECKPOINT_INTERVAL_MS = 500.0
E6_CHECKPOINT_RESTORE_MS = 500.0
E6_GOSSIP_DETECT_MS = 1500.0
E6_MULTILAYER_DETECTION_MS = 500.0
E6_CLUSTER_DETECT_MS = 500.0
E6_CLUSTER_SELECTION_MS = 100.0
E6_CLUSTER_REROUTE_MS = 150.0
E6_ACK_KEY_BYTES = 28

E7_N = 100
E7_NODE = "F2"
E7_NET_SENSOR_TO_FOG_MS = 2.0
E7_NET_SENSOR_TO_CLOUD_MS = 30.0

# E7 ANALYTICAL PIPELINE MODEL — hardware-target constants
# =========================================================
# E7 is NOT a Python-execution benchmark. It is an analytical latency model
# using hardware-target values (AES hardware accelerators, SGX enclave timing,
# Paillier co-processor estimates). Python interpreter overhead measured in E2
# is 20–50x higher because it runs unoptimised Python big-integer arithmetic.
# E7 and E2 are intentionally different: E2 measures the Python prototype;
# E7 models the target deployment hardware. E7_MODEL_NOTE is stored in every
# E7 CSV row so reviewers see the distinction without reading source code.
E7_MODEL_NOTE = (
    "E7 uses analytical hardware-target latency constants (see config.py). "
    "These are NOT comparable to E2 Python-measured latencies, which are 20–50x "
    "higher due to unoptimised Python big-integer arithmetic."
)
E7_SENSOR_AES_MS = 0.001        # hardware AES accelerator (e.g. STM32 AES peripheral)
E7_PLAINTEXT_SUM_MS = 1.0       # fog CPU plaintext accumulation, n=100 readings
E7_PAILLIER_ENC_MS = 3.8        # hardware-target Paillier encryption per slot
E7_PAILLIER_ADD_MS = 0.01       # Paillier homomorphic addition (ciphertext-space op)
E7_KMM_COMBINE_MS = 1.0         # KMM combine step inside TEE
E7_STORAGE_PREP_MS = 5.0        # enclave storage serialisation
E7_CLOUD_UPLOAD_MS = 10.0       # network upload to cloud store
E7_TEE_DELEGATION_MS = 150.0    # TEE key delegation round-trip (provision + ACK)
E7_METHODS = ["cloud_only", "fog_plaintext", "paillier_fog_convert", "ours"]
E7_STAGE_COLUMNS = [
    "sensor_to_fog_ms",
    "sensor_to_cloud_ms",
    "sensor_aes_ms",
    "plaintext_sum_ms",
    "enclave_aes_to_paillier_ms",
    "paillier_accum_ms",
    "delegation_ms",
    "kmm_combine_ms",
    "storage_prep_ms",
    "cloud_upload_ms",
]

E8_SCENARIOS = [
    "one_fog_compromised",
    "backup_during_delegation",
    "kmm_compromised",
    "host_os_reads_enclave",
]

FOG_NODES = {
    "F1": {
        "cpu_cap": 4,
        "ram_gb": 8,
        "bw_mbps": 100,
        "latency": 0.2,
        "sensors": 20,
        "class": "Strong",
        "speed_factor": 1.0,
    },
    "F2": {
        "cpu_cap": 2,
        "ram_gb": 4,
        "bw_mbps": 50,
        "latency": 0.4,
        "sensors": 20,
        "class": "Medium",
        "speed_factor": 1.8,
    },
    "F3": {
        "cpu_cap": 1,
        "ram_gb": 2,
        "bw_mbps": 20,
        "latency": 0.8,
        "sensors": 20,
        "class": "Weak",
        "speed_factor": 3.2,
    },
    "F4": {
        "cpu_cap": 2,
        "ram_gb": 4,
        "bw_mbps": 80,
        "latency": 0.2,
        "sensors": 20,
        "class": "Medium-Fast",
        "speed_factor": 1.4,
    },
    "F5": {
        "cpu_cap": 4,
        "ram_gb": 8,
        "bw_mbps": 100,
        "latency": 0.3,
        "sensors": 20,
        "class": "Strong",
        "speed_factor": 1.0,
    },
}

BASE_WORKLOAD = {
    "F1": {"workload": 0.20, "latency": 0.30, "queue": 0.20},
    "F2": {"workload": 0.50, "latency": 0.40, "queue": 0.40},
    "F3": {"workload": 0.60, "latency": 0.80, "queue": 0.55},
    "F4": {"workload": 0.40, "latency": 0.10, "queue": 0.30},
    "F5": {"workload": 0.45, "latency": 0.30, "queue": 0.40},
}


def build_schedule() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window in range(1, E5_WINDOWS + 1):
        if window <= 5:
            f2_load, f5_alive, event = 0.50, True, "Normal"
        elif window <= 10:
            f2_load, f5_alive, event = 0.85, True, "F2 overload"
        elif window == 11:
            f2_load, f5_alive, event = 0.50, False, "F5 fails"
        elif window <= 13:
            f2_load, f5_alive, event = 0.50, False, "Gossip detects F5"
        elif window <= 17:
            f2_load, f5_alive, event = 0.50, True, "Recovery"
        else:
            f2_load, f5_alive, event = 0.90, True, "F2 second overload"
        rows.append(
            {
                "window": window,
                "f2_load": f2_load,
                "f5_alive": f5_alive,
                "event": event,
            }
        )
    return rows


# Node speed factors (phi) — canonical values matching paper Table (node definitions).
# phi=1.0 is fastest (F1, F5); phi=0.25 is slowest (F3).
# Used in compat_HP to structurally penalise slow nodes for compute-heavy tasks.
NODE_PHI: dict[str, float] = {
    "F1": 1.00,
    "F2": 0.50,
    "F3": 0.25,
    "F4": 0.50,
    "F5": 1.00,
}

# S6 Full-CapScore weight vectors (w1, w2, w3, w4); w1+w2+w3+w4 = 1.0
# w4 = compat weight (dominant at 0.45 in both task types)
FULLCAPSCORE_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    "LS": (0.10, 0.35, 0.10, 0.45),
    "HP": (0.15, 0.30, 0.10, 0.45),
}


def _compat(node_id: str, node: dict[str, float], task_type: str) -> float:
    """Structural compatibility term for Full-CapScore (S6).

    Reads individual utilisation fractions U_cpu/U_ram/U_bw from the node
    entry when available; falls back to the composite workload scalar otherwise
    (e.g. when called with legacy BASE_WORKLOAD entries that predate S6).
    """
    u_cpu = node.get("U_cpu", node.get("workload", 0.5))
    u_ram = node.get("U_ram", node.get("workload", 0.5))
    u_bw  = node.get("U_bw",  node.get("workload", 0.5))
    if task_type == "LS":
        # BW-dominant: sensor readings need bandwidth for retransmission
        return 0.10 * (1.0 - u_cpu) + 0.10 * (1.0 - u_ram) + 0.80 * (1.0 - u_bw)
    else:
        # HP: phi^2 penalty steers compute-heavy tasks away from slow nodes
        phi = NODE_PHI.get(node_id, 0.5)
        return 0.60 * phi**2 * (1.0 - u_cpu) + 0.30 * (1.0 - u_ram) + 0.10 * (1.0 - u_bw)


def full_cap_score(
    node_id: str,
    workload_table: dict[str, dict[str, float]],
    task_type: str,
) -> float:
    """Full-CapScore (S6): four-term task-aware fog node scoring formula.

    score = w1*(1-workload) + w2*(1-latency) + w3*(1-queue) + w4*compat(node, task_type)

    task_type: "LS" (Latency-Sensitive) or "HP" / any non-LS string (Heavy Processing).
    Raw sensor ciphertexts routed during fault recovery use "LS" by default.
    """
    node = workload_table[node_id]
    # Treat any non-LS type (including legacy "TO") as HP
    tt = "LS" if task_type == "LS" else "HP"
    w1, w2, w3, w4 = FULLCAPSCORE_WEIGHTS[tt]
    return (
        w1 * (1.0 - node["workload"])
        + w2 * (1.0 - node["latency"])
        + w3 * (1.0 - node["queue"])
        + w4 * _compat(node_id, node, tt)
    )
