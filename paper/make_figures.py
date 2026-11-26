"""Figures for the one-page evidence note, generated only from committed results and the frozen config.

Run from the repository root:  python paper/make_figures.py
Inputs : results/summary/summary.json, results/summary/bootstrap.json, configs/experiment.toml
Outputs: paper/fig1_seed_accuracy.svg, paper/fig2_gap_forest.svg
Figures are drawn at their final printed size (1:1), so font sizes below are the sizes on the page.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
MM = 1 / 25.4

FULL = "#0072B2"      # Full-561      (validated categorical pair: CVD dE 21.9, normal dE 31.2)
COMPACT = "#D55E00"   # Compact-100
INK, MUTED, GRID = "#1a1a1a", "#4d4d4d", "#dcdcdc"
FS = 7.0              # on-page font size (pt)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.3,
    "ytick.labelsize": FS - 0.3, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "svg.fonttype": "path",
    "legend.fontsize": FS - 0.3,
    "svg.hashsalt": "gao2026-har-reproduction",  # deterministic element IDs
})
SVG_META = {"Date": None}  # no build timestamp: identical inputs give byte-identical SVGs


def load():
    summary = json.loads((ROOT / "results/summary/summary.json").read_text(encoding="utf-8"))
    boot = json.loads((ROOT / "results/summary/bootstrap.json").read_text(encoding="utf-8"))
    with open(ROOT / "configs/experiment.toml", "rb") as fh:
        pub = tomllib.load(fh)["published"]
    return summary, boot, pub


def fig1(summary, pub, width_mm=98, height_mm=41):
    rows = summary["per_seed"]
    seeds = [r["seed"] for r in rows]
    x = list(range(len(seeds)))
    full = [100 * r["full561_accuracy"] for r in rows]
    comp = [100 * r["compact100_accuracy"] for r in rows]
    fig, ax = plt.subplots(figsize=(width_mm * MM, height_mm * MM))
    # Published values: dashed reference lines in the colour of the corresponding model.
    ax.axhline(100 * pub["full561_accuracy"], color=FULL, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.axhline(100 * pub["compact100_accuracy"], color=COMPACT, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.plot(x, full, "o", ms=4.8, color=FULL, mec="white", mew=0.7, zorder=3)
    ax.plot(x, comp, "s", ms=4.4, color=COMPACT, mec="white", mew=0.7, zorder=3)
    # Five-seed means: short horizontal bars to the right of the seeds.
    xm = 4.75
    for vals, col in ((full, FULL), (comp, COMPACT)):
        m = sum(vals) / len(vals)
        ax.plot([xm - 0.17, xm + 0.17], [m, m], color=col, lw=2.0, solid_capstyle="butt", zorder=3)
    ax.text(4.5, 100 * pub["full561_accuracy"] + 0.06, f"published full: {100 * pub['full561_accuracy']:.2f}%",
            ha="right", va="bottom", fontsize=FS - 0.5)
    ax.text(4.5, 100 * pub["compact100_accuracy"] + 0.06,
            f"published compact: {100 * pub['compact100_accuracy']:.2f}%", ha="right", va="bottom", fontsize=FS - 0.5)
    ax.set_xticks(x + [xm], [f"{s}\n(ref.)" if s == summary["reference_seed"] else str(s) for s in seeds] + ["5-seed\nmean"])
    ax.set_xlim(-0.45, 5.1)
    ax.set_ylim(93.2, 96.0)
    ax.set_yticks([93.5, 94.0, 94.5, 95.0, 95.5])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("Training seed", labelpad=2)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [Line2D([], [], ls="", marker="o", ms=4.8, color=FULL, mec="white", mew=0.7, label="Full-561"),
               Line2D([], [], ls="", marker="s", ms=4.4, color=COMPACT, mec="white", mew=0.7,
                      label="Compact-100 (X-first)"),
               Line2D([], [], color=INK, lw=0.9, ls=(0, (4, 2)), label="Published value")]
    # One-row legend in the empty band between the two series (covers no data point).
    leg = ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.45, 0.555), ncol=3, frameon=True,
                    handletextpad=0.4, borderaxespad=0.2, columnspacing=1.1, borderpad=0.3, framealpha=1.0,
                    handlelength=1.6)
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_facecolor("white")
    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.27, top=0.985)
    fig.savefig(OUT / "fig1_seed_accuracy.svg", metadata=SVG_META)
    plt.close(fig)


def fig2(summary, boot, pub, width_mm=80, height_mm=41):
    ref = next(r for r in summary["per_seed"] if r["seed"] == summary["reference_seed"])
    gaps = summary["five_seed"]["gap_pp"]
    fig, ax = plt.subplots(figsize=(width_mm * MM, height_mm * MM))
    rows = [(2, "Published", pub["gap_pp"], pub["bootstrap_ci_pp"], "s"),
            (1, f"This work,\nseed {summary['reference_seed']}", ref["gap_pp"], boot["ci_pp"], "o")]
    ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for y, _, pt, ci, mk in rows:
        ax.plot(ci, [y, y], color=INK, lw=1.2, solid_capstyle="butt", zorder=2)
        for c in ci:
            ax.plot([c, c], [y - 0.11, y + 0.11], color=INK, lw=1.2, zorder=2)
        ax.plot([pt], [y], mk, ms=5.0, color=INK, mec="white", mew=0.7, zorder=3)
        ax.text(pt, y + 0.2, f"{pt:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]", ha="center", va="bottom",
                fontsize=FS - 0.5)
    # Five individual seed gaps and their mean; no interval is drawn for the mean.
    ax.plot([gaps["mean"], gaps["mean"]], [-0.2, 0.2], color=INK, lw=1.6, solid_capstyle="butt", zorder=2)
    ax.plot(gaps["values"], [0] * len(gaps["values"]), "o", ms=3.6, mfc="white", mec=INK, mew=0.8, zorder=3)
    ax.text(gaps["mean"], 0.24, f"mean {gaps['mean']:+.2f} (no CI)", ha="center", va="bottom", fontsize=FS - 0.5)
    ax.set_yticks([2, 1, 0], [rows[0][1], rows[1][1], "This work,\n5 seeds"])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(-0.45, 2.62)
    ax.set_xlim(-0.6, 3.6)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel(r"$\Delta$ = Acc$_\mathrm{full}$ $-$ Acc$_\mathrm{compact}$ (pp)", labelpad=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.215, right=0.985, bottom=0.21, top=0.985)
    fig.savefig(OUT / "fig2_gap_forest.svg", metadata=SVG_META)
    plt.close(fig)


def main():
    summary, boot, pub = load()
    fig1(summary, pub)
    fig2(summary, boot, pub)
    print("wrote paper/fig1_seed_accuracy.svg, paper/fig2_gap_forest.svg")


if __name__ == "__main__":
    main()
