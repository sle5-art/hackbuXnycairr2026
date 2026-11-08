"""
Slide figures for SensorGap, kept out of the live map so the demo stays map-only.

Writes outputs/slides/:
    reliability_curve.png   copy of the strict-validation curve (made by python -m sensorgap.validate --strict)
    placement_table.png     the 10 suggested sensor sites
    fairness.png            95% band half-width before -> after the 10 sites, high-asthma tracts vs the rest

Run from the repo root after labels/placement/validate:
    python -m app.slide_figures
"""

import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.dashboard import BOROUGH, OUT, PIN, UNIT

SLIDES = OUT / "slides"
INK, MUTED, GRID = "#1d1a24", "#6b6575", "#e2dee9"


def placement_table(placement, path):
    t = placement.assign(borough=placement["tract"].str[2:5].map(BOROUGH))
    rows = [[str(r.rank), r.borough, r.tract, f"{r.asthma_pct:.1f}%", f"{r.adults:,}",
             "yes" if pd.notna(r.rank_if_population_only) else ""] for r in t.itertuples()]
    cols = ["Pin", "Borough", "Census tract", "Adult asthma", "Adults", "Population-only\nalso picks it"]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.1), dpi=200)
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=cols, cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1],
                   colWidths=[0.07, 0.17, 0.22, 0.16, 0.12, 0.2])
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.visible_edges = "B"
        cell.get_text().set_color(INK if i else MUTED)
        if i == 0:
            cell.get_text().set_fontweight("bold")
        if j == 0 and i:
            cell.get_text().set_color(PIN)
            cell.get_text().set_fontweight("bold")
    ax.set_title("Where 10 new sensors cut uncertainty most for adults with asthma", loc="left", color=INK,
                 fontsize=12, fontweight="bold", pad=10)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fairness(summary, path):
    groups = [("Highest-asthma quarter of tracts", summary["band_halfwidth_high_asthma_tracts"], PIN),
              ("All other tracts", summary["band_halfwidth_other_tracts"], "#8a8494")]
    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=200)
    for y, (name, (before, after), color) in enumerate(reversed(groups)):
        ax.plot([before, after], [y, y], color=color, lw=2, solid_capstyle="round", zorder=1)
        ax.scatter([before], [y], s=70, facecolor="white", edgecolor=color, lw=2, zorder=2)
        ax.scatter([after], [y], s=70, color=color, edgecolor="white", lw=2, zorder=3)
        ax.annotate(f"±{before:.2f}", (before, y), xytext=(0, 10), textcoords="offset points", ha="center",
                    color=MUTED, fontsize=9)
        ax.annotate(f"±{after:.2f}", (after, y), xytext=(0, 10), textcoords="offset points", ha="center",
                    color=INK, fontsize=9, fontweight="bold")
        ax.text(-0.02, y, name, transform=ax.get_yaxis_transform(), ha="right", va="center", color=INK, fontsize=10)
    lo = min(min(v) for _, v, _ in groups)
    hi = max(max(v) for _, v, _ in groups)
    pad = (hi - lo) * 0.25 or 0.05
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.6, len(groups) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel(f"Half-width of the 95% band ({UNIT}), open = before, filled = after {summary['k']} new sensors",
                  color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("The new sensors narrow the gap for the tracts with the most adult asthma", loc="left",
                 color=INK, fontsize=12, fontweight="bold")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    SLIDES.mkdir(parents=True, exist_ok=True)
    made = []
    if (OUT / "reliability_curve.png").exists():
        shutil.copy(OUT / "reliability_curve.png", SLIDES / "reliability_curve.png")
        made.append("reliability_curve.png")
    if (OUT / "placement.csv").exists():
        placement_table(pd.read_csv(OUT / "placement.csv", dtype={"tract": str}), SLIDES / "placement_table.png")
        made.append("placement_table.png")
    if (OUT / "placement_summary.json").exists():
        fairness(json.loads((OUT / "placement_summary.json").read_text()), SLIDES / "fairness.png")
        made.append("fairness.png")
    print(f"Wrote {', '.join(made) or 'nothing (run validate/placement first)'} to {SLIDES}")


if __name__ == "__main__":
    main()
