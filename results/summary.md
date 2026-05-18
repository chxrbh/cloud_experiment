# Repaired P1 Experiment Summary

This summary is generated from script-produced CSV files, not notebook state.

Run mode: full configured experiment; key_bits=2048; Paillier backend=phe.

## E2 Latency Conclusion
All tested n values are within the 500 ms aggregation window.

Interpretation: E2 uses the calibrated 2048-bit host-reference benchmark for the proposed path. The calibrated mean total is 444.128 ms, within the 500 ms window. Caveat: 2048-bit host reference benchmark; OP-TEE QEMU timing only, not physical hardware timing. TA-side 2048-bit Paillier is not enabled in this QEMU prototype.

Security note: SGX/TEE behavior is simulated; E3-style correctness checks are not security proofs, and hardware isolation remains a formal assumption.
