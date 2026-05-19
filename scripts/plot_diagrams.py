"""Generate architecture and workflow diagrams for main.tex.

Produces:
  figures/overall_diagram.png / .pdf  — 6-layer IIoT edge–fog architecture
  figures/flow_diagram.png    / .pdf  — secure data-processing flowchart
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

import plot_style

plot_style.apply()

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def rbox(ax, x, y, w, h, text, facecolor, fontsize=7, va="center"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06",
        facecolor=facecolor,
        edgecolor="#2C3E50",
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va=va,
        fontsize=fontsize,
        multialignment="center",
        zorder=3,
    )


def diamond(ax, cx, cy, w, h, text, facecolor, fontsize=6.5):
    pts = np.array([
        [cx,       cy + h / 2],
        [cx + w / 2, cy],
        [cx,       cy - h / 2],
        [cx - w / 2, cy],
    ])
    poly = plt.Polygon(pts, facecolor=facecolor, edgecolor="#2C3E50", linewidth=1.2, zorder=2)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            multialignment="center", zorder=3)


def arrow(ax, x1, y1, x2, y2, label="", label_side="right"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=1.1),
        zorder=4,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx = 0.12 if label_side == "right" else -0.12
        ax.text(mx + dx, my, label, ha="center", va="center",
                fontsize=5.5, color="#555555")


def bidir_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="<|-|>", color="#2C3E50", lw=1.1), zorder=4)


# ---------------------------------------------------------------------------
# Figure 1 — overall_diagram
# ---------------------------------------------------------------------------

def plot_architecture():
    fig, ax = plt.subplots(figsize=(plot_style.IEEE_DOUBLE, 4.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # Layer definitions: (x, y, w, h, color, label)
    LAYER_X, LAYER_W = 0.4, 7.2
    LAYER_H = 0.85
    GAP = 0.28

    layers = [
        ("#AED6F1", "IIoT Sensor Layer\n(100 sensors, AES-128-GCM, 4 readings / 500 ms)"),
        ("#A9DFBF", "Edge Gateway Layer\n(AES-GCM tag verification; forward ciphertext)"),
        ("#F9E79F", "Fog SGX Enclave Layer\n(AES → Paillier, sealed $k_{fogI}$, remote attestation)"),
        ("#FAD7A0", "Fog Paillier Aggregation Layer\n(running $C_{agg,F_i}$, per-reading homomorphic update)"),
        ("#F1948A", "Fog Load-Balancing Layer\n(gossip $T_g{=}500\\,$ms, CapacityScore delegation)"),
        ("#D7BDE2", "Cloud Storage\n($C_{data}$: AES-GCM encrypted aggregate)"),
    ]

    ys = []
    for i, (color, label) in enumerate(layers):
        y = i * (LAYER_H + GAP) + 0.3
        ys.append(y)
        rbox(ax, LAYER_X, y, LAYER_W, LAYER_H, label, facecolor=color, fontsize=6.8)

    # Up-arrows between layers
    for i in range(len(layers) - 1):
        y_bot = ys[i] + LAYER_H
        y_top = ys[i + 1]
        arrow(ax, LAYER_X + LAYER_W / 2, y_bot, LAYER_X + LAYER_W / 2, y_top)

    # KMM box — aligned with fog layers (indices 2, 3, 4)
    kmm_y0 = ys[2] - 0.05
    kmm_y1 = ys[4] + LAYER_H + 0.05
    kmm_x = LAYER_X + LAYER_W + 0.35
    kmm_w = 2.35
    rbox(ax, kmm_x, kmm_y0, kmm_w, kmm_y1 - kmm_y0,
         "Key Management\nModule (KMM)\n• key provisioning\n• slot map\n• window combine",
         facecolor="#D5D8DC", fontsize=6.5)

    # Bidirectional arrows: fog layer sides → KMM
    for i in [2, 3, 4]:
        cy = ys[i] + LAYER_H / 2
        bidir_arrow(ax, LAYER_X + LAYER_W, cy, kmm_x, cy)

    ax.set_title(
        "IIoT Edge–Fog Architecture: Six-Layer Pipeline with KMM Trust Anchor",
        fontsize=plot_style.FS_TITLE, pad=4,
    )
    fig.tight_layout(pad=0.3)
    out_base = os.path.join(FIGURES_DIR, "overall_diagram")
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)
    print(f"  saved {out_base}.png / .pdf")


# ---------------------------------------------------------------------------
# Figure 2 — flow_diagram
# ---------------------------------------------------------------------------

def plot_workflow():
    fig, ax = plt.subplots(figsize=(plot_style.IEEE_DOUBLE, 6.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10.8)
    ax.axis("off")

    CX = 5.5          # center x for main column
    BW, BH = 4.2, 0.72  # box width / height
    DW, DH = 3.8, 0.92  # diamond width / height

    # ---- main column (top to bottom) ----
    # [0] sensor encrypt
    y0 = 9.8
    rbox(ax, CX - BW / 2, y0, BW, BH,
         "Sensor: AES-128-GCM encrypt\n$C_i^{AES}=Enc_{GCM}(k_{fogI}, n, m_i, A)$",
         "#AED6F1")

    arrow(ax, CX, y0, CX, y0 - 0.28)

    # [1] edge gateway
    y1 = y0 - 0.28 - BH
    rbox(ax, CX - BW / 2, y1, BW, BH,
         "Edge Gateway: verify AES-GCM tag\nreject $C_i^{AES}$ if tag fails",
         "#A9DFBF")

    arrow(ax, CX, y1, CX, y1 - 0.28)

    # [2] decision: workload >= tau?
    y2 = y1 - 0.28 - DH
    diamond(ax, CX, y2 + DH / 2, DW, DH,
            "Workload$(F_A) \\geq \\tau$?",
            "#F9E79F", fontsize=7)

    # ---- NO path (left) ----
    left_cx = 2.1
    arrow(ax, CX - DW / 2, y2 + DH / 2, left_cx + 1.6, y2 + DH / 2, label="No", label_side="left")

    y_fa = y2 + DH / 2 - 0.36 - BH
    rbox(ax, left_cx - 1.6, y_fa, 3.2, BH,
         "Fog A enclave:\nDec AES → Enc Paillier\n$C_{agg,F_A} \\leftarrow C_{agg,F_A} \\oplus C_i^P$",
         "#FAD7A0", fontsize=6.5)
    arrow(ax, CX - DW / 2, y2 + DH / 2, left_cx, y2 + DH / 2 - 0.01)
    arrow(ax, left_cx, y2 + DH / 2 - 0.01, left_cx, y_fa + BH)

    # ---- YES path (right) ----
    right_cx = 8.9
    arrow(ax, CX + DW / 2, y2 + DH / 2, right_cx - 1.3, y2 + DH / 2, label="Yes", label_side="right")

    y_zf = y2 + DH / 2 - 0.36 - BH
    rbox(ax, right_cx - 1.3, y_zf, 2.6, BH,
         "Fog A: zero-fill slot\n$C_{agg,F_A} \\leftarrow C_{agg,F_A} \\oplus C_{zero}$",
         "#F1948A", fontsize=6.5)
    arrow(ax, right_cx, y2 + DH / 2 - 0.01, right_cx, y_zf + BH)

    y_kmm = y_zf - 0.28 - BH
    rbox(ax, right_cx - 1.3, y_kmm, 2.6, BH,
         "KMM: provision $k_{fogA}$\nto attested Enclave B",
         "#D7BDE2", fontsize=6.5)
    arrow(ax, right_cx, y_zf, right_cx, y_kmm + BH)

    y_fb = y_kmm - 0.28 - BH
    rbox(ax, right_cx - 1.3, y_fb, 2.6, BH,
         "Fog B enclave:\nDec AES → Enc Paillier\n$C_{agg,F_B} \\leftarrow C_{agg,F_B} \\oplus C_s^P$",
         "#FAD7A0", fontsize=6.5)
    arrow(ax, right_cx, y_kmm, right_cx, y_fb + BH)

    # ---- merge at window close ----
    y_merge = y_fa - 0.42
    # left path down
    arrow(ax, left_cx, y_fa, left_cx, y_merge + 0.18)
    # right path down
    arrow(ax, right_cx, y_fb, right_cx, y_merge + 0.18)
    # converge arrows to center
    ax.annotate("", xy=(CX, y_merge + 0.18), xytext=(left_cx, y_merge + 0.18),
                arrowprops=dict(arrowstyle="-", color="#2C3E50", lw=1.1), zorder=4)
    ax.annotate("", xy=(CX, y_merge + 0.18), xytext=(right_cx, y_merge + 0.18),
                arrowprops=dict(arrowstyle="-", color="#2C3E50", lw=1.1), zorder=4)
    arrow(ax, CX, y_merge + 0.18, CX, y_merge - 0.02)

    y_wc = y_merge - 0.02 - BH
    rbox(ax, CX - BW / 2, y_wc, BW, BH,
         "KMM window close ($W{=}500\\,$ms):\n"
         "$C_{agg}^{final}=C_{agg,F_1}\\oplus\\cdots\\oplus C_{agg,F_k}$",
         "#D5D8DC")

    arrow(ax, CX, y_wc, CX, y_wc - 0.28)

    y_sp = y_wc - 0.28 - BH
    rbox(ax, CX - BW / 2, y_sp, BW, BH,
         "Storage-prep enclave: Dec Paillier, AES-GCM encrypt\n"
         "$C_{data}=Enc_{AES}(k_{store},\\{D_{agg},\\mathrm{meta}\\})$",
         "#D7BDE2")

    arrow(ax, CX, y_sp, CX, y_sp - 0.28)

    y_cloud = y_sp - 0.28 - BH
    rbox(ax, CX - BW / 2, y_cloud, BW, BH,
         "Cloud: store $C_{data}$ (one object per window)",
         "#AED6F1")

    ax.set_title(
        "Secure IIoT Data-Processing Workflow: TEE Delegation, Paillier Slot Aggregation, KMM Window Combine",
        fontsize=plot_style.FS_TITLE, pad=4,
    )
    fig.tight_layout(pad=0.3)
    out_base = os.path.join(FIGURES_DIR, "flow_diagram")
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)
    print(f"  saved {out_base}.png / .pdf")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating architecture diagrams...")
    plot_architecture()
    plot_workflow()
    print("Done.")
