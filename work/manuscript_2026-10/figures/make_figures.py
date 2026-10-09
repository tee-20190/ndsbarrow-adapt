"""Manuscript figures 1-8 from saved aggregate RESULT files only. No maps, no coordinates, no entity or lead
locations. Run in the WSL venv that has matplotlib 3.11 (the v3venv has no matplotlib):

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
      /home/tee-/adaf-h2h-venv/bin/python -B work/manuscript_2026-10/figures/make_figures.py

v6 (2026-10-09): files numbered in order of first citation in DRAFT_v6 (F1 design, F2 block bootstrap, F3 main
result, F4 recall vs budget, F5 Stage 2, F6 pool terms, F7 convention sensitivity, F8 IGNORE readings); labels use
the paper's terms; F7 has no connecting lines; F8 prints every bar value, including 0. v5 outputs are kept as *_v5.*.
v7 (2026-10-09, REFEREE_v6 P5/P6): F1 notes the earlier legacy scoring of the ten runs in Verden and, under
Oldenburg, the development of benchmark v0 on earlier systems' saved proposals; F4 and F5 plot recall as a
fraction, as in the text and tables; "seeds" -> "runs" in F1 and F5 labels. v6 F1, F4, F5 kept as *_v6.*.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S1 = ROOT / "work/stage1_d1_v0_2026-10-07/RESULT.json"
S2 = ROOT / "work/stage2_d1_controls_2026-10-07/RESULT.json"
VD = ROOT / "work/d1_verden_descriptive_2026-10-07/RESULT.json"
OT = ROOT / "work/oldenburg_transfer_2026-10-07/RESULT.json"
IGN = ROOT / "work/gpt6_pro_collab_2026-10-07/v0_uncertainty/IGNORE_V0.json"
D1IG = HERE.parent / "ignore_rescore/D1_IGNORE_RESCORE.json"
B = ("0.5", "1.0", "2.0")
DEC = ("RAW", "NMS60")
ADAF = "ADAF_barrow"
D1 = tuple(f"D1F1_BARROW_L_{s}" for s in range(2026092805, 2026092815))
CTL = tuple(f"S2CTL_BARROW_{s}" for s in range(2026092805, 2026092815))
C_ADAF, C_D1, C_CTL, C_GREY = "#4c72b0", "#dd8452", "#8172b3", "#7f7f7f"


def rj(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(HERE / f"{name}.png", dpi=200)
    fig.savefig(HERE / f"{name}.pdf")
    plt.close(fig)


def contrasts() -> dict:
    """district -> decoder -> {point, joint, per_run} (fractions)."""
    vd = rj(VD)["grids"]["2km"]["decoders"]
    ot = rj(OT)["bootstrap"]["2km"]["decoders"]
    return {"Verden": {d: vd[d]["contrast"] for d in DEC}, "Oldenburg": {d: ot[d] for d in DEC}}


def f1_design() -> None:
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.axis("off")
    ax.set_ylim(-0.12, 1.0)
    steps = [("Training districts (3)\n852 + 852 windows\ntraining only", "#e8e8e8"),
             ("Verden\nrecipe selection\n(development seeds 02-04)", "#f6d7c3"),
             ("Stage 1 (Verden)\nD1 vs ADAF barrow\n10 runs; frozen criterion", "#f6d7c3"),
             ("Stage 2 (Verden)\nD1 vs replay control\npaired; frozen criterion", "#f6d7c3"),
             ("Oldenburg\none-time transfer check\nscored once", "#c9d9ef")]
    notes = {2:"rule frozen before v0 scoring;\nthe ten runs were scored here\nearlier (legacy metric)",
             3: "rule frozen before scoring",
             4: "rule frozen before any D1\noutput there; v0 developed\npartly on earlier systems'\nsaved proposals here"}
    w, gap = 0.16, 0.035
    for i, (txt, col) in enumerate(steps):
        x = 0.01 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.3), w, 0.5, boxstyle="round,pad=0.01", fc=col, ec="black", lw=0.8))
        ax.text(x + w / 2, 0.55, txt, ha="center", va="center", fontsize=7.5)
        if i:
            ax.annotate("", xy=(x - 0.003, 0.55), xytext=(x - gap + 0.003, 0.55),
                        arrowprops={"arrowstyle": "->", "lw": 0.8})
        if i in notes:
            ax.text(x + w / 2, 0.22, notes[i], ha="center", va="top", fontsize=6.5, style="italic")
    ax.text(0.5, 0.92, "Dataset roles and sequence (no terrain shared between training and evaluation districts)",
            ha="center", fontsize=8.5)
    save(fig, "F1_design")


def f2_main(c: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (dist, cc) in zip(axes, c.items()):
        for j, d in enumerate(DEC):
            for i, b in enumerate(B):
                x = i + (j - 0.5) * 0.36
                pr = [100 * cc[d]["per_run"][n][b] for n in D1]
                ax.scatter(np.full(len(pr), x) + np.linspace(-0.06, 0.06, len(pr)), pr, s=8,
                           color=C_D1 if d == "RAW" else C_GREY, alpha=0.7, zorder=2)
                lo, hi = (100 * v for v in cc[d]["ci95"]["joint"][b])
                pt = 100 * cc[d]["point"][b]
                ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt="s", color="black", ms=4, capsize=3, zorder=3)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(3), [f"b = {float(b):g}" for b in B])
        ax.set_title(dist, fontsize=9)
    axes[0].set_ylabel("D1 − ADAF barrow recall (points)")
    axes[1].scatter([], [], color=C_D1, s=8, label="per run, RAW (left of pair)")
    axes[1].scatter([], [], color=C_GREY, s=8, label="per run, NMS60 (right of pair)")
    axes[1].errorbar([], [], fmt="s", color="black", label="ten-run mean, joint 95 % (2 km)")
    axes[1].legend(fontsize=6.5, loc="upper left")
    save(fig, "F3_main_result")


def f3_stage2() -> None:
    s = rj(S2)["systems"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, d in zip(axes, DEC):
        for i, b in enumerate(B):
            a = s[ADAF][d][b]["recall"]
            ax.hlines(a, i - 0.35, i + 0.35, color=C_ADAF, lw=2, label="ADAF barrow" if i == 0 else None)
            for k, (names, col, off, lab) in enumerate(((D1, C_D1, -0.12, "D1 (10 runs)"),
                                                        (CTL, C_CTL, 0.12, "replay control (10 runs)"))):
                v = [s[n][d][b]["recall"] for n in names]
                ax.scatter(np.full(10, i + off) + np.linspace(-0.05, 0.05, 10), v, s=10, color=col,
                           label=lab if i == 0 else None)
        ax.set_xticks(range(3), [f"b = {float(b):g}" for b in B])
        ax.set_title(f"Verden, {d}", fontsize=9)
    axes[0].set_ylabel("registered-barrow recall")
    axes[0].legend(fontsize=6.5)
    save(fig, "F5_stage2_three_way")


def f4_budget() -> None:
    s1, ot = rj(S1)["systems"], rj(OT)["exact_scorer"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (dist, sy) in zip(axes, (("Verden", s1), ("Oldenburg", ot))):
        x = [float(b) for b in B]
        c = [sy[ADAF]["RAW"][b]["ceiling_c"] for b in B]
        ax.plot(x, c, "k--", lw=1, label="budget ceiling c = min(1, K/N)")
        ax.plot(x, [sy[ADAF]["RAW"][b]["recall"] for b in B], "o-", color=C_ADAF, label="ADAF barrow")
        for n in D1:
            ax.plot(x, [sy[n]["RAW"][b]["recall"] for b in B], "-", color=C_D1, alpha=0.35, lw=0.8)
        ax.plot([], [], "-", color=C_D1, label="D1 runs")
        k2 = sy[ADAF]["RAW"]["2.0"]
        if k2.get("slots_filled", k2["K"]) < k2["K"]:
            ax.annotate(f"ADAF barrow fills {k2['slots_filled']} of {k2['K']} slots", (2, k2["recall"]),
                        xytext=(1.15, 0.10), fontsize=6.5, arrowprops={"arrowstyle": "->"})
        ax.set_xticks(x, ["0.5", "1", "2"])
        ax.set_xlabel("budget b (proposals per km², all proposals charged)")
        ax.set_title(f"{dist}, RAW", fontsize=9)
    axes[0].set_ylabel("registered-barrow recall")
    axes[0].legend(fontsize=6.5)
    save(fig, "F4_recall_vs_budget")


def f5_pool() -> None:
    vd, ot = rj(VD)["pool_terms"], rj(OT)["pool_terms"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), sharey="row")
    for col, (dist, pt) in enumerate((("Verden", vd), ("Oldenburg", ot))):
        for row, d in enumerate(DEC):
            ax = axes[row, col]
            names = (ADAF,) + D1
            m = np.array([pt[n][d]["budgets"]["0.5"]["M"] for n in names])
            gr = np.array([pt[n][d]["budgets"]["0.5"]["G_ranking"] for n in names])
            gp = np.array([pt[n][d]["G_pool"] for n in names])
            other = np.array([pt[n][d]["C"] for n in names]) - m - gr
            x = np.arange(len(names))
            ax.bar(x, m, color="#55a868", label="M (matched at b = 0.5)")
            ax.bar(x, gr, bottom=m, color="#c44e52", label="G_ranking (recoverable, unranked)")
            ax.bar(x, other, bottom=m + gr, color="#ccb974", label="C beyond min(K, C)")
            ax.bar(x, gp, bottom=m + gr + other, color="#d0d0d0", label="G_pool (no ranking recovers)")
            ax.set_xticks(x, ["ADAF\nbarrow"] + [n[-2:] for n in D1], fontsize=6.5)
            ax.set_title(f"{dist}, {d}", fontsize=9)
    axes[0, 0].set_ylabel("registered barrows")
    axes[1, 0].set_ylabel("registered barrows")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(HERE / "F6_pool_terms.png", dpi=200)
    fig.savefig(HERE / "F6_pool_terms.pdf")
    plt.close(fig)


def f6_sensitivity() -> None:
    # detector_week1/RESULT.md:241-245 (points; primary = mean over three budgets)
    # Row labels are Table 5's wording; conventions are categories, so markers are not joined (referee C3).
    rows = {"Oldenburg: earlier adaptation − its equal-update control": (14.1, 6.7, 4.1, 6.0),
            "Oldenburg: earlier adaptation − ADAF barrow": (4.5, 1.5, 0.4, -0.2),
            'Verden: earlier fused project system − ADAF "all objects"': (10.2, 5.9, 3.5, 1.3),
            "Verden: earlier fused project system − ADAF barrow": (-15.6, -10.8, -8.5, -14.7)}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for k, (lab, v) in enumerate(rows.items()):
        ax.scatter(np.arange(4) + (k - 1.5) * 0.12, v, marker="osD^"[k], s=28, color=f"C{k}", label=lab, zorder=3)
    ax.axhline(0, color="black", lw=0.6)
    for x in (0.5, 1.5, 2.5):
        ax.axvline(x, color="#d0d0d0", lw=0.6)
    ax.set_xticks(range(4), ["legacy group\n(original protocol)", "legacy\nindividual", "v0 NMS60", "v0 RAW"])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("contrast, mean over three budgets (points)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False)
    save(fig, "F7_convention_sensitivity")


def f7_ignore() -> None:
    ig, d1 = rj(IGN)["regions"], rj(D1IG)["regions"]
    lab = {"historical": "(i)", "literal": "(ii)", "precedence": "(iii)"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    dists = ("verden", "oldenburg")
    w = 0.25
    for k, v in enumerate(lab):
        vals = [100 * ig[d]["variants"][v]["unreachable_entities"] / d1[d]["N"] for d in dists]
        bars = axes[0].bar(np.arange(2) + (k - 1) * w, vals, w, label=lab[v])
        # Print every value, so the zero bars of (i) and (iii) read as 0, not as missing (referee C2).
        axes[0].bar_label(bars, labels=[f"{x:.0f}" for x in vals], fontsize=7, padding=2)
        for j, d in enumerate(dists):
            r = d1[d]["readings"][v]["decoders"]["RAW"]
            pt = 100 * r["diff_point"]["0.5"]
            lo, hi = (100 * x for x in r["diff_joint_ci95"]["0.5"])
            axes[1].errorbar(j + (k - 1) * w, pt, yerr=[[pt - lo], [hi - pt]], fmt="o", capsize=3,
                             color=f"C{k}", label=lab[v] if j == 0 else None)
    for ax in axes:
        ax.set_xticks(range(2), ["Verden", "Oldenburg"])
    axes[0].set_ylabel("eligible barrows unreachable (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_ylabel("D1 − ADAF barrow, RAW b = 0.5 (points)")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[0].legend(fontsize=7, title="IGNORE reading", title_fontsize=7, loc="upper center", ncol=3)
    save(fig, "F8_ignore_readings")


def f8_blocks() -> None:
    """Synthetic illustration only: random points on a toy grid; no real data."""
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((i * 2, j * 2), 2, 2, fill=False, lw=0.6, ec="grey"))
    refs = rng.uniform(0.3, 9.7, size=(14, 2))
    refs[0] = (3.95, 5.1)  # one reference whose 15 m reach straddles a cell edge
    ax.scatter(refs[:, 0], refs[:, 1], s=14, color=C_D1, zorder=3, label="register entity (synthetic)")
    ax.add_patch(Rectangle((2, 4), 4, 2, fill=True, alpha=0.18, color=C_ADAF, label="cells merged by reach"))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Block bootstrap: 2 km cells; cells joined when an entity's\n15 m reach crosses an edge (toy example)",
                 fontsize=8)
    ax.legend(fontsize=6.5, loc="lower right")
    save(fig, "F2_block_bootstrap_illustration")


def main() -> None:
    c = contrasts()
    f1_design()
    f2_main(c)
    f3_stage2()
    f4_budget()
    f5_pool()
    f6_sensitivity()
    f7_ignore()
    f8_blocks()
    print(sorted(p.name for p in HERE.glob("*.png")))


if __name__ == "__main__":
    main()
