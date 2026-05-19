"""Unit and experiment-shape tests for the repaired P1 package."""

from __future__ import annotations

import random
import tempfile
import unittest

from config import (
    CALIBRATION_E2_TOTAL_MS,
    CALIBRATION_HOST_AGGREGATE_MEAN_MS,
    CALIBRATION_PAILLIER_CIPHERTEXT_BYTES,
    E2_N_VALUES,
    E3_K_VALUES,
    E3_N,
    E3_TRIALS,
    E3B_TRIALS,
    E4_K_VALUES,
    E5_N_WINDOWS,
    E5_SEEDS,
    E5_STRATEGY_NAMES,
    E5_TASK_PARAMS,
    E6_FAILURE_SCENARIOS,
    E6_METHODS,
    E6_SEEDS,
    E6_CLUSTER_DETECT_MS,
    E6_CLUSTER_REROUTE_MS,
    E6_CLUSTER_SELECTION_MS,
    E6_MULTILAYER_DETECTION_MS,
    E7_METHODS,
    E7_STAGE_COLUMNS,
    E8_SCENARIOS,
    KMM_PROV_MS,
    SCALE,
    T_ACK_MS,
    full_cap_score,
)
from crypto_sim import (
    KMM,
    aes_decrypt,
    aes_encrypt,
    generate_fog_keys,
    generate_paillier_keypair,
    paillier_encrypt,
    sgx_enclave_process,
)
from experiments_p1 import (
    _compute_boundaries,
    aggregate_results,
    run_e1,
    run_e2,
    run_e3a,
    run_one_seed,
)
from experiments_p2 import run_e3b, run_e4, run_e6
from experiments_p3 import run_e7, run_e8
from results import validate_p2_results, write_csv


class P1RepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(7)
        self.pub, self.priv = generate_paillier_keypair(256, self.rng)
        self.k_fog, self.k_store = generate_fog_keys(["F1", "F2", "F3", "F4", "F5"], self.rng)

    def test_aes_round_trip(self) -> None:
        nonce, ct = aes_encrypt(self.k_fog["F1"], 12.345, self.rng)
        self.assertEqual(aes_decrypt(self.k_fog["F1"], nonce, ct), 12.345)

    def test_paillier_scaled_addition(self) -> None:
        a = paillier_encrypt(self.pub, 12 * SCALE, self.rng)
        b = paillier_encrypt(self.pub, 7 * SCALE, self.rng)
        self.assertEqual(self.priv.decrypt(a + b), 19 * SCALE)

    def test_kmm_combine(self) -> None:
        kmm = KMM(self.k_fog)
        c1 = paillier_encrypt(self.pub, 4, self.rng)
        c2 = paillier_encrypt(self.pub, 9, self.rng)
        combined, _ = kmm.combine({"F1": c1, "F4": c2}, self.pub)
        self.assertEqual(self.priv.decrypt(combined), 13)

    def test_key_provision_revoke(self) -> None:
        kmm = KMM(self.k_fog)
        key, latency = kmm.provision_key("F1", "F4")
        self.assertEqual(key, self.k_fog["F1"])
        self.assertEqual(latency, 50.0)
        self.assertIn("F1", kmm.delegated["F4"])
        kmm.revoke_key("F1", "F4")
        self.assertNotIn("F1", kmm.delegated["F4"])

    def test_e1_formulas(self) -> None:
        rows = run_e1(self.pub)
        row100 = next(row for row in rows if row["n"] == 100)
        self.assertEqual(row100["ours_ciphertexts"], 1)
        self.assertEqual(row100["ciphertext_count_reduction"], 100)
        self.assertAlmostEqual(row100["byte_reduction_vs_aes"], row100["aes_bytes"] / row100["ours_cloud_aes_bytes"])
        self.assertEqual(row100["ours_cloud_aes_bytes"], 36,
            "AES-GCM per-reading: 12 nonce + 8 body (worst-case ASCII float) + 16 tag")

    def test_e2_shape_quick(self) -> None:
        rows = run_e2(self.pub, self.priv, self.k_fog, self.k_store, seed=7, reps=1, n_values=[10])
        self.assertEqual([row["n"] for row in rows], [10])
        for key in ["plaintext_ms_median", "aes_ms_median", "paillier_nobatch_ms_median", "ours_ms_median"]:
            self.assertIn(key, rows[0])
        self.assertAlmostEqual(rows[0]["ours_ms_median"], CALIBRATION_E2_TOTAL_MS)
        self.assertTrue(rows[0]["ours_within_500ms"])
        self.assertEqual(rows[0]["paillier_ciphertext_bytes"], CALIBRATION_PAILLIER_CIPHERTEXT_BYTES)

    def test_e3a_shape_quick_and_config(self) -> None:
        self.assertEqual(E3_N, 100)
        self.assertEqual(E3_TRIALS, 100)
        self.assertEqual(E3_K_VALUES, [1, 5, 10, 20, 50])
        rows = run_e3a(self.pub, self.priv, self.k_fog, self.k_store, seed=7, trials=2, n=10, k_values=[1])
        self.assertEqual(rows[0]["accuracy_scaled_pct"], 100.0)
        self.assertIn("median_quantization_error", rows[0])

    def test_e5_shape_and_config(self) -> None:
        self.assertEqual(E5_N_WINDOWS, 20)
        self.assertEqual(E5_SEEDS, list(range(42, 62)))
        self.assertEqual(E5_STRATEGY_NAMES, ["S1", "S2", "S3", "S4", "S5", "S6"])
        self.assertEqual(E5_TASK_PARAMS["LS"]["max_delay"], 100.0)
        self.assertEqual(E5_TASK_PARAMS["HP"]["max_delay"], 500.0)

        window_starts, _ = _compute_boundaries()
        per_seed = []
        for strategy in E5_STRATEGY_NAMES:
            metrics, _records, _snaps, _sigma = run_one_seed(strategy, 42, window_starts)
            per_seed.append(metrics)
        aggregate = aggregate_results(per_seed)
        self.assertEqual({row["strategy"] for row in aggregate}, set(E5_STRATEGY_NAMES))
        for row in aggregate:
            self.assertIn("deadline_rate_mean", row)
            self.assertIn("sigma_bw_mean_mean", row)
            self.assertIn("t_wait_mean_mean", row)
        s6 = next(row for row in aggregate if row["strategy"] == "S6")
        self.assertGreater(float(s6["deadline_rate_mean"]), 0.70)

    def test_e3b_multisource_shape_and_edges(self) -> None:
        rows = run_e3b(self.pub, self.priv, self.k_fog, self.k_store, seed=7, trials=2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(E3B_TRIALS, 100)
        self.assertEqual(rows[0]["accuracy_scaled_pct"], 100.0)
        self.assertEqual(rows[0]["provisioned_edges"], "F1->F4;F2->F4")

    def test_e4_kmm_combine_shape_and_bytes(self) -> None:
        rows = run_e4(self.pub, seed=7, reps=1, k_values=[1, 2, 5])
        self.assertEqual([row["k_fog_aggregates"] for row in rows], [1, 2, 5])
        for row in rows:
            self.assertEqual(row["he_additions"], row["k_fog_aggregates"] - 1)
            self.assertEqual(row["bytes_received"], row["k_fog_aggregates"] * row["ciphertext_bytes"])
            self.assertAlmostEqual(row["combine_latency_ms_median"], CALIBRATION_HOST_AGGREGATE_MEAN_MS)
            self.assertEqual(row["ciphertext_bytes"], CALIBRATION_PAILLIER_CIPHERTEXT_BYTES)
        self.assertEqual(E4_K_VALUES, [1, 2, 5, 10, 20, 50, 100])

    def test_p2_result_validation_rejects_mixed_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_csv(
                f"{tmp}/metadata_p2.csv",
                [
                    {"key": "seed", "value": 42},
                    {"key": "key_bits", "value": 2048},
                    {"key": "quick_smoke_run", "value": False},
                ],
            )
            write_csv(f"{tmp}/e3b_multisource_correctness.csv", [{"trials": 2}])
            write_csv(f"{tmp}/e4_kmm_combine.csv", [{"ciphertext_bytes": 64}])
            write_csv(f"{tmp}/e6_fault_detection.csv", [{"method": "proposed_ack_kmm"}])
            with self.assertRaises(ValueError):
                validate_p2_results(tmp, quick=False, key_bits=2048, expected_ciphertext_bytes=512)

    def test_e6_fault_model_pairs_and_ack_bound(self) -> None:
        rows = run_e6()
        self.assertEqual(len(rows), len(E6_FAILURE_SCENARIOS) * len(E6_SEEDS) * len(E6_METHODS))
        self.assertEqual(
            set(E6_METHODS),
            {"b1_gossip", "b2_replication", "checkpoint", "b4_multilayer", "b5_fog_clustering", "proposed_ack_kmm"},
        )
        ack_rows = [row for row in rows if row["method"] == "proposed_ack_kmm"]
        for row in ack_rows:
            self.assertEqual(row["recovery_latency_ms"], T_ACK_MS + KMM_PROV_MS)
            self.assertEqual(row["total_readings"], row["recovered_readings"] + row["lost_readings"])
            self.assertAlmostEqual(row["data_loss_rate"], row["lost_readings"] / row["total_readings"])
            self.assertAlmostEqual(row["message_overhead"], row["measured_message_units"] / row["baseline_message_units"])
            self.assertAlmostEqual(row["compute_overhead"], row["measured_compute_ops"] / row["baseline_compute_ops"])
        before = [row for row in rows if row["scenario"] == "fail_0ms" and row["seed"] == 0]
        gossip = next(row for row in before if row["method"] == "b1_gossip")
        replication = next(row for row in before if row["method"] == "b2_replication")
        checkpoint = next(row for row in before if row["method"] == "checkpoint")
        multilayer = next(row for row in before if row["method"] == "b4_multilayer")
        clustering = next(row for row in before if row["method"] == "b5_fog_clustering")
        ack = next(row for row in before if row["method"] == "proposed_ack_kmm")
        self.assertEqual(replication["recovery_latency_ms"], 0.0)
        self.assertEqual(checkpoint["recovery_latency_ms"], 500.0)
        self.assertEqual(gossip["recovery_latency_ms"], 1500.0)
        self.assertEqual(multilayer["baseline_id"], "B4")
        self.assertEqual(multilayer["recovery_latency_ms"], E6_MULTILAYER_DETECTION_MS)
        self.assertEqual(clustering["baseline_id"], "B5")
        self.assertEqual(
            clustering["recovery_latency_ms"],
            E6_CLUSTER_DETECT_MS + E6_CLUSTER_SELECTION_MS + E6_CLUSTER_REROUTE_MS,
        )
        self.assertGreater(replication["message_overhead"], ack["message_overhead"])
        self.assertGreater(replication["compute_overhead"], checkpoint["compute_overhead"])
        self.assertGreater(checkpoint["checkpoint_writes"], 0)
        self.assertGreater(checkpoint["data_loss_rate"], 0.0,
            "fail_0ms checkpoint must lose readings during the 500ms restore window")
        self.assertGreater(ack["control_messages"], 0)
        self.assertLess(multilayer["data_loss_rate"], gossip["data_loss_rate"])
        self.assertGreater(multilayer["compute_overhead"], gossip["compute_overhead"])
        self.assertGreater(gossip["data_loss_rate"], clustering["data_loss_rate"])
        self.assertGreater(clustering["data_loss_rate"], multilayer["data_loss_rate"])
        self.assertEqual(replication["lost_readings"], 0)
        mid_window = [row for row in rows if row["scenario"] == "mid_window_250ms" and row["seed"] == 0]
        mid_checkpoint = next(row for row in mid_window if row["method"] == "checkpoint")
        mid_ack = next(row for row in mid_window if row["method"] == "proposed_ack_kmm")
        self.assertGreater(mid_checkpoint["data_loss_rate"], mid_ack["data_loss_rate"])

    def test_e7_pipeline_shape_totals_and_storage(self) -> None:
        rows, summary = run_e7(
            self.pub,
            self.priv,
            self.k_fog,
            self.k_store,
            seed=7,
            n=5,
            windows=2,
        )
        self.assertEqual(len(rows), 2 * len(E7_METHODS))
        self.assertEqual({row["method"] for row in rows}, set(E7_METHODS))
        self.assertEqual({row["method"] for row in summary}, set(E7_METHODS))
        for row in rows:
            for column in E7_STAGE_COLUMNS:
                self.assertIn(column, row)
            stage_total = sum(float(row[column]) for column in E7_STAGE_COLUMNS)
            self.assertAlmostEqual(float(row["total_ms"]), stage_total, places=9)
        storage_by_method = {row["method"]: row["storage_bytes_per_window"] for row in rows if row["window"] == 1}
        self.assertGreater(storage_by_method["cloud_only"], storage_by_method["fog_plaintext"])
        self.assertEqual(storage_by_method["paillier_fog_convert"], storage_by_method["ours"],
            "paillier_fog_convert now stores 1 aggregate; distinction vs ours is latency not storage")
        self.assertNotEqual(storage_by_method["fog_plaintext"], storage_by_method["ours"])
        items_by_method = {row["method"]: row["storage_items_per_window"] for row in rows if row["window"] == 1}
        self.assertEqual(items_by_method["cloud_only"], 5)
        self.assertEqual(items_by_method["fog_plaintext"], 1)
        self.assertEqual(items_by_method["paillier_fog_convert"], 1)
        self.assertEqual(items_by_method["ours"], 1)

    def test_e8_blast_radius(self) -> None:
        rows = run_e8()
        self.assertEqual([row["scenario"] for row in rows], E8_SCENARIOS)
        by_scenario = {row["scenario"]: row for row in rows}
        one_fog = by_scenario["one_fog_compromised"]
        self.assertEqual(one_fog["global_key_exposed_pct"], 100.0)
        self.assertEqual(one_fog["fog_scoped_exposed_pct"], 20.0)
        backup = by_scenario["backup_during_delegation"]
        self.assertEqual(backup["fog_scoped_exposed_pct"], 40.0)
        kmm = by_scenario["kmm_compromised"]
        self.assertEqual(kmm["global_key_exposed_pct"], 100.0)
        self.assertEqual(kmm["fog_scoped_exposed_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
