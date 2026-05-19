"""Shared visual style for all experiment figures.

Call apply() once at the top of any plotting module to enforce consistent
fonts, colors, line widths, and markers across every figure in the paper.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# IEEE figure widths (inches)
# ---------------------------------------------------------------------------
IEEE_SINGLE = 3.45
IEEE_DOUBLE = 7.16

# ---------------------------------------------------------------------------
# Color palette — canonical; use these keys throughout all plot scripts
# "yours" (dark green) = proposed scheme, always visually dominant
# ---------------------------------------------------------------------------
C: dict[str, str] = {
    "yours":    "#1B7F4F",  # proposed — dark green (print-safe)
    "aes":      "#4878CF",  # AES baseline — blue
    "paillier": "#D62728",  # Paillier no-batch — red
    "plain":    "#AAAAAA",  # plaintext / weakest baseline — grey
    "rr":       "#9467BD",  # round-robin — purple
    "thresh":   "#FF7F0E",  # threshold-only — orange
    "random":   "#AAAAAA",  # random — same grey as plain
}

DEADLINE_COLOR = "#CC2222"  # deadline / budget reference lines

# ---------------------------------------------------------------------------
# Markers and line widths
# ---------------------------------------------------------------------------
MARKERS: dict[str, str] = {
    "yours":    "D",   # diamond — most distinctive
    "aes":      "o",
    "paillier": "^",
    "plain":    "s",
    "rr":       "v",
    "thresh":   "P",
    "random":   "s",
}

LW_PROPOSED = 2.2
LW_BASELINE = 1.4

# ---------------------------------------------------------------------------
# Storage / latency baselines — E1, E2, E7
# Keys match CSV column name prefixes and method strings
# ---------------------------------------------------------------------------
SL_LABELS: dict[str, str] = {
    "plaintext":              "Plaintext (insecure)",
    "aes_per_reading":        "AES per reading",
    "paillier_nobatch":       "Paillier (no batch)",
    "ours":                   "Proposed (slot agg.)",
    "cloud_only":             "Cloud-only",
    "fog_plaintext":          "Fog plaintext",
    "paillier_fog_convert":   "Paillier fog-convert",
}

SL_COLORS: dict[str, str] = {
    "plaintext":              C["plain"],
    "aes_per_reading":        C["aes"],
    "paillier_nobatch":       C["paillier"],
    "ours":                   C["yours"],
    "cloud_only":             C["plain"],
    "fog_plaintext":          C["aes"],
    "paillier_fog_convert":   C["paillier"],
}

SL_MARKERS: dict[str, str] = {
    "plaintext":              "s",
    "aes_per_reading":        "o",
    "paillier_nobatch":       "^",
    "ours":                   "D",
    "cloud_only":             "s",
    "fog_plaintext":          "o",
    "paillier_fog_convert":   "^",
}

# ---------------------------------------------------------------------------
# Load-balancing strategies — E5
# ---------------------------------------------------------------------------
E5_COLORS: dict[str, str] = {
    "S1": "#D62728",   # red
    "S2": "#FF7F0E",   # orange
    "S3": "#BCBD22",   # yellow-green
    "S4": "#1F77B4",   # blue
    "S5": "#2CA02C",   # green
    "S6": C["yours"],  # dark green — proposed CapacityScore
}

# ---------------------------------------------------------------------------
# Fault-tolerance methods — E6
# ---------------------------------------------------------------------------
FT_ORDER: list[str] = [
    "b1_gossip", "b2_replication", "checkpoint",
    "b4_multilayer", "b5_fog_clustering", "proposed_ack_kmm",
]

FT_LABELS: dict[str, str] = {
    "b1_gossip":          "B1 Gossip",
    "b2_replication":     "B2 Replication",
    "checkpoint":         "B3 Checkpoint",
    "b4_multilayer":      "B4 Multilayer",
    "b5_fog_clustering":  "B5 Fog-Cluster",
    "proposed_ack_kmm":   "Proposed\n(ACK+KMM)",
}

FT_COLORS: dict[str, str] = {
    "b1_gossip":          "#888888",
    "b2_replication":     C["paillier"],
    "checkpoint":         C["thresh"],
    "b4_multilayer":      C["aes"],
    "b5_fog_clustering":  C["rr"],
    "proposed_ack_kmm":   C["yours"],
}

FT_MARKERS: dict[str, str] = {
    "b1_gossip":          "o",
    "b2_replication":     "s",
    "checkpoint":         "^",
    "b4_multilayer":      "D",
    "b5_fog_clustering":  "P",
    "proposed_ack_kmm":   "H",
}

# Backward-compat aliases (figures.py uses these names)
E6_METHOD_LABELS = FT_LABELS
E6_METHOD_COLORS = FT_COLORS

# ---------------------------------------------------------------------------
# Pipeline / method labels (E7)
# ---------------------------------------------------------------------------
METHOD_LABELS: dict[str, str] = {
    "random":               "Random",
    "round_robin":          "Round-Robin",
    "threshold":            "Threshold only",
    "capacity":             "CapacityScore (ours)",
    "cloud_only":           "Cloud-only",
    "fog_plaintext":        "Fog plaintext",
    "paillier_fog_convert": "Paillier\nfog convert",
    "ours":                 "Proposed",
}

METHOD_COLORS: dict[str, str] = {
    "random":      C["random"],
    "round_robin": C["rr"],
    "threshold":   C["thresh"],
    "capacity":    C["yours"],
}

# ---------------------------------------------------------------------------
# Typography constants (reference these instead of inline fontsize=N)
# ---------------------------------------------------------------------------
FS_SUPTITLE = 8.5
FS_TITLE    = 8
FS_LABEL    = 8
FS_TICK     = 7
FS_LEGEND   = 7
FS_ANNOT    = 6.5

# ---------------------------------------------------------------------------
# apply() — call once per script to set global rcParams
# ---------------------------------------------------------------------------
def apply() -> None:
    """Apply the paper-wide IEEE style to matplotlib rcParams."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "font.size":          FS_TICK,
        "axes.titlesize":     FS_TITLE,
        "axes.labelsize":     FS_LABEL,
        "xtick.labelsize":    FS_TICK,
        "ytick.labelsize":    FS_TICK,
        "legend.fontsize":    FS_LEGEND,
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "#cccccc",
        "lines.linewidth":    LW_BASELINE,
        "lines.markersize":   4.5,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.5,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    })


def deadline_line(ax, y: float, label: str = "500 ms window") -> None:
    """Draw a consistent deadline / budget reference line."""
    ax.axhline(y, color=DEADLINE_COLOR, linestyle="--", lw=1.2, alpha=0.8, label=label)


def strip_spines(ax) -> None:
    """Remove top and right spines (idempotent helper)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
