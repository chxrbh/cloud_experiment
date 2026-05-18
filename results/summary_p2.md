# Repaired P2 Experiment Summary

This summary is generated from script-produced CSV files, not notebook state.

Run mode: full configured experiment; key_bits=2048; Paillier backend=phe.

Implemented P2 experiments: E3b multi-source correctness, E4 KMM combine overhead, and E6 ACK/KMM fault recovery.

E4 note: KMM combine uses the calibrated 2048-bit host aggregate reference (1.474 ms). Caveat: 2048-bit host reference benchmark; OP-TEE QEMU timing only, not physical hardware timing. TA-side 2048-bit Paillier is not enabled in this QEMU prototype.

E3b note: multi-source correctness is a scaled-sum arithmetic sanity check; it is not a cryptographic security proof or an SGX isolation experiment.

E6 note: fault recovery is an analytical reading-loss model comparing B1 gossip-based detection, B2 replication-based fault tolerance, B3 checkpoint/restart, B4 multilayer detection, B5 fog-clustering fault tolerance, and proposed ACK+KMM.
