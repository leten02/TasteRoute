"""Draw the model, ranking and routing figures from results/*.json.

Style matches data_analysis_preprocessing/figures: conclusion as the title,
the data source as the subtitle, one y-axis, light horizontal grid, direct
labels instead of legend boxes where possible.

Usage:  python experiments/figures.py   (after the experiments have run)
Output: figures/*.png
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")  # matplotlib.use("Agg"): render to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

# reference categorical palette, fixed order (blue, orange, aqua), and text inks
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})


def load(name):
    """Read results/<name>.json."""
    return json.loads((RES / f"{name}.json").read_text())


def titled(fig, title, subtitle):
    """Add the conclusion title and the source subtitle above the plot."""
    fig.text(0.01, 1.035, title, ha="left", va="bottom", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.995, subtitle, ha="left", va="bottom", fontsize=10, color=INK2)


def save(fig, name):
    """Write figures/<name>.png and close the figure."""
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def v1_rmse():
    """V1: test RMSE per model with 95% bootstrap intervals."""
    d = load("model_compare")["warm"]
    names = list(d)
    rm = np.array([d[n]["rmse"] for n in names])
    lo = rm - np.array([d[n]["rmse_ci95"][0] for n in names])
    hi = np.array([d[n]["rmse_ci95"][1] for n in names]) - rm
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [GRID if n in ("Global mean", "Item-kNN") else BLUE for n in names]
    ax.bar(names, rm, color=colors, width=0.6, yerr=[lo, hi], ecolor=INK2, capsize=3)
    for x, v in enumerate(rm):
        ax.text(x, v + hi[x] + 0.004, f"{v:.3f}", ha="center", fontsize=10, color=INK)
    ax.set_ylim(1.0, 1.22)
    ax.set_ylabel("Test RMSE (lower is better)")
    titled(fig, "Bias, MF, CBF and Hybrid tie at RMSE 1.05",
           "Warm test users, 69,847 ratings · bars: 95% bootstrap interval over users")
    save(fig, "V1_rmse_by_model")


def v5_cold():
    """V5: cold-start RMSE at 1, 3 and 5 training ratings (RQ2)."""
    d = load("model_compare")["cold"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ms = [1, 3, 5]
    # Bias, MF, CBF and Hybrid differ by < 0.002, so they share one line
    for name, label, color in [("Bias", "Bias = MF = CBF = Hybrid", BLUE), ("Item-kNN", "Item-kNN", ORANGE)]:
        ys = [d[name][str(m)] for m in ms]
        ax.plot(ms, ys, color=color, lw=2, marker="o", ms=6)
        ax.text(5.15, ys[-1], label, va="center", color=INK)
    ax.set_xticks(ms, [f"{m} rating{'s' if m > 1 else ''}" for m in ms])
    ax.set_xlim(0.7, 7.2)
    ax.set_ylabel("Test RMSE")
    titled(fig, "CBF does not yet help new users (RQ2)",
           "Cold-start users keep m training ratings · Bias, MF, CBF and Hybrid lie within 0.002")
    save(fig, "V5_cold_start")


def m1_knn():
    """M1: item-kNN validation RMSE against k per similarity."""
    g = load("item_knn")["grid"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for sim, color in [("cosine", BLUE), ("adjusted", ORANGE), ("pearson", AQUA)]:
        pts = [x for x in g if x["similarity"] == sim and x["beta"] == 0]
        ax.plot([p["k"] for p in pts], [p["val_rmse"] for p in pts], color=color, lw=2, marker="o", ms=6)
        nudge = {"adjusted": 0.004, "pearson": -0.004}.get(sim, 0)
        ax.text(84, pts[-1]["val_rmse"] + nudge, sim, va="center", color=INK)
    ax.axhline(1.0443, color=INK2, lw=1, ls="--")
    ax.text(10, 1.047, "Bias model 1.044", color=INK2, fontsize=10)
    ax.set_xticks([10, 20, 40, 80])
    ax.set_xlim(5, 100)
    ax.set_xlabel("Neighbors k")
    ax.set_ylabel("Validation RMSE")
    titled(fig, "Item-kNN stays above the bias model at every k",
           "Shrinkage beta = 0 (best) · more neighbors help, but users rate a median of 6 places")
    save(fig, "M1_knn_grid")


def m3_mf_curve():
    """M3: MF train and validation RMSE per epoch."""
    c = load("mf")["curve"]
    ep = [h["epoch"] for h in c]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, [h["train_rmse"] for h in c], color=BLUE, lw=2)
    ax.plot(ep, [h["val_rmse"] for h in c], color=ORANGE, lw=2)
    ax.text(ep[-1] + 0.3, c[-1]["train_rmse"], "train", va="center", color=INK)
    ax.text(ep[-1] + 0.3, c[-1]["val_rmse"], "validation", va="center", color=INK)
    best = min(c, key=lambda h: h["val_rmse"])
    ax.plot(best["epoch"], best["val_rmse"], "o", color=ORANGE, ms=8)
    ax.annotate(f"stop: epoch {best['epoch']}", (best["epoch"], best["val_rmse"]),
                xytext=(0, 14), textcoords="offset points", ha="center", color=INK, fontsize=10)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_xlim(0.5, ep[-1] + 3)
    titled(fig, "MF overfits after 15 epochs; early stopping keeps the best",
           "f = 10, lambda = 0.1, eta = 0.01 · training stops after 3 epochs without improvement")
    save(fig, "M3_mf_learning_curve")


def m4_mf_grid():
    """M4: MF validation RMSE over factors f and lambda (heatmap)."""
    g = load("mf")["grid"]
    fs = sorted({x["f"] for x in g})
    lams = sorted({x["lam"] for x in g})
    Z = np.array([[next(x["val_rmse"] for x in g if x["f"] == f and x["lam"] == l) for l in lams] for f in fs])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(False)
    # sequential single hue: darker = lower (better) RMSE
    ax.imshow(-Z, cmap="Blues", aspect="auto")
    for a in range(len(fs)):
        for b in range(len(lams)):
            dark = -Z[a, b] > (-Z).mean()
            ax.text(b, a, f"{Z[a, b]:.4f}", ha="center", va="center", fontsize=10, color="white" if dark else INK)
    ax.set_xticks(range(len(lams)), [str(l) for l in lams])
    ax.set_yticks(range(len(fs)), [str(f) for f in fs])
    ax.set_xlabel("Regularization lambda")
    ax.set_ylabel("Latent factors f")
    titled(fig, "Fewest factors and strongest regularization win",
           "MF validation RMSE · darker is better · little structure beyond the biases")
    save(fig, "M4_mf_grid")


def m5_blend():
    """M5/M6: hybrid validation RMSE against alpha and kappa."""
    d = load("cbf_hybrid")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    a1.plot([x["alpha"] for x in d["hybrid_alpha"]["curve"]], [x["val_rmse"] for x in d["hybrid_alpha"]["curve"]],
            color=BLUE, lw=2, marker="o", ms=5)
    a1.set_xlabel("Fixed CF weight alpha")
    a1.set_ylabel("Validation RMSE")
    a2.plot([x["kappa"] for x in d["hybrid_kappa"]["curve"]], [x["val_rmse"] for x in d["hybrid_kappa"]["curve"]],
            color=BLUE, lw=2, marker="o", ms=5)
    a2.set_xscale("log")
    a2.set_xlabel("Adaptive kappa (log scale)")
    titled(fig, "The blend weight barely matters",
           "Hybrid of MF and CBF · both curves span less than 0.0015 RMSE")
    save(fig, "M5_M6_hybrid_weight")


def r2_pareto():
    """R2: NDCG@10 against mean list distance per decay and tau (RQ1)."""
    d = load("ranking")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for decay, color in [("exp", BLUE), ("gaussian", ORANGE), ("cutoff", AQUA)]:
        pts = [x for x in d["rerank"] if x["decay"] == decay]
        ax.plot([p["mean_dist_km"] for p in pts], [p["NDCG@10"] for p in pts], color=color, lw=2, marker="o", ms=5)
        at = max(pts, key=lambda p: p["NDCG@10"]) if decay == "cutoff" else max(pts, key=lambda p: p["mean_dist_km"])
        ax.text(at["mean_dist_km"] + 0.1, at["NDCG@10"] + (0.0002 if decay == "cutoff" else 0), decay,
                va="center", color=INK)
    base = d["models"]["Hybrid"]
    ax.plot(base["mean_dist_km"], base["NDCG@10"], "o", color=INK, ms=8)
    ax.annotate(f"no distance\n{base['mean_dist_km']:.1f} km", (base["mean_dist_km"], base["NDCG@10"]),
                xytext=(10, 0), textcoords="offset points", ha="left", va="center", color=INK, fontsize=10)
    tau2 = next(x for x in d["rerank"] if x["decay"] == "exp" and x["tau"] == 2)
    ax.annotate(f"exp, tau = 2 km\n{tau2['mean_dist_km']:.1f} km", (tau2["mean_dist_km"], tau2["NDCG@10"]),
                xytext=(12, -4), textcoords="offset points", color=INK, fontsize=10)
    ax.set_xlabel("Mean distance of the top-10 list from the user's anchor (km)")
    ax.set_ylabel("NDCG@10")
    ax.set_xlim(1.5, 7.8)
    titled(fig, "Closer lists are also more accurate (RQ1)",
           "Hybrid candidates re-ranked by distance decay, one point per scale tau · 3,000 test users")
    save(fig, "R2_accuracy_vs_distance")


def t2_timing():
    """T2: route solver runtime against N (log scale)."""
    t = load("route")["timing"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label, color in [("brute_force", "brute force", ORANGE), ("held_karp", "Held-Karp", BLUE),
                              ("nn_2opt", "NN + 2-opt", AQUA)]:
        pts = [(r["N"], r[key]) for r in t if key in r]
        ax.plot(*zip(*pts), color=color, lw=2, marker="o", ms=5)
        ax.text(pts[-1][0] + 0.2, pts[-1][1], label, va="center", color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("Places N")
    ax.set_ylabel("Runtime per route (ms, log scale)")
    ax.set_xlim(2.5, 14)
    titled(fig, "Held-Karp solves a day route exactly in about 1 ms",
           "Mean of 10 random instances per N · brute force stops at N = 9")
    save(fig, "T2_route_runtime")


def t3_gap():
    """T3: how much longer each ordering is than the optimal route (RQ3)."""
    rows = load("route")["per_user"]
    opt = np.array([r["optimal"] for r in rows])
    gaps = {
        "recommendation\norder": np.array([r["rec_order"] for r in rows]) / opt - 1,
        "nearest\nneighbor": np.array([r["nn"] for r in rows]) / opt - 1,
        "NN + 2-opt": np.array([r["nn_2opt"] for r in rows]) / opt - 1,
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([100 * g for g in gaps.values()], showfliers=False, widths=0.5,
               medianprops={"color": BLUE, "lw": 2}, boxprops={"color": INK2}, whiskerprops={"color": INK2},
               capprops={"color": INK2})
    ax.set_xticks(range(1, 4), list(gaps))
    for x, g in enumerate(gaps.values(), start=1):
        ax.text(x + 0.3, 100 * np.median(g), f"median {100 * np.median(g):.0f}%", va="center", fontsize=10, color=INK)
    ax.set_ylabel("Longer than the optimal route (%)")
    titled(fig, "Visiting in recommendation order walks 26% more",
           "3,000 test users, 6 places each · NN + 2-opt is exactly optimal for 82% of users")
    save(fig, "T3_route_gap")


def d1_popularity():
    """D1: Precision@10 of popularity vs the popularity-blended hybrid."""
    pop = load("popularity")
    d = pop["test"]
    # gamma chosen on validation by experiments/popularity.py (same rule), never re-picked on test
    gamma = max((v for v in pop["val"] if v["method"] == "blend"), key=lambda v: v["NDCG@10"])["param"]
    keys = ["popularity(-)", "popularity(-) + distance", f"blend({gamma})", f"blend({gamma}) + distance"]
    labels = ["Popularity", "Popularity\n+ distance", "Hybrid blend", "Hybrid blend\n+ distance"]
    vals = [d[k]["P@10"] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color=[GRID, GRID, BLUE, BLUE], width=0.6)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.0002, f"{v:.4f}", ha="center", fontsize=10, color=INK)
    raw = load("ranking")["models"]["Hybrid"]["P@10"]
    ax.axhline(raw, color=INK2, lw=1, ls="--")
    ax.text(-0.3, raw + 0.0003, f"Hybrid alone {raw:.4f}", ha="left", color=INK2, fontsize=10)
    ax.set_ylabel("Precision@10")
    titled(fig, "Popularity and distance carry the ranking signal",
           f"Hybrid blended with log popularity (gamma = {gamma}, chosen on validation) vs popularity alone "
           "· 3,000 test users")
    save(fig, "D1_popularity_blend")


def l1_learned():
    """L1: NDCG@10 of the fixed distance rule vs the learned visit model (issue #48)."""
    d = load("learned_distance")
    test = d["test"]
    fixed200 = next(k for k in test if k.startswith("blend + fixed distance (top 200"))
    keys = ["popularity + fixed distance", "blend + fixed distance (top 50)", fixed200,
            "learned (no personal radius)", "learned (with personal radius)"]
    labels = ["Popularity\n+ rule", "Blend + rule\ntop 50", "Blend + rule\ntop 200",
              "Learned\nno radius", "Learned\n+ radius"]
    vals = [test[k]["NDCG@10"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(labels, vals, color=[GRID, GRID, GRID, ORANGE, ORANGE], width=0.6)
    for x, k in enumerate(keys):
        ax.text(x, vals[x] + 0.0004, f"{vals[x]:.4f}", ha="center", fontsize=10, color=INK)
        ax.text(x, 0.0012, f"cov {test[k]['coverage'] * 100:.2f}%", ha="center", fontsize=9, color=INK2)
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0, max(vals) * 1.15)
    p = d["paired_vs_fixed"][fixed200]["learned (with personal radius)"]["p"]
    titled(fig, "A learned distance model ties the tuned rule with more variety",
           f"Logistic regression on distance, travel radius, rating, popularity · vs tuned rule p = {p:.2f} "
           "· cov = catalog coverage · 3,000 test users")
    save(fig, "L1_learned_vs_fixed")


def l2_distance_curve():
    """L2: the learned distance penalty in popularity terms, by travel radius."""
    m = load("learned_distance")["model"]
    c = m["curve"]
    km = c["km"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (radius, ys), color in zip(c["popularity_needed"].items(), [ORANGE, INK2, BLUE]):
        ax.plot(km, ys, color=color, lw=2)
        ax.text(km[-1] + 0.15, ys[-1], f"radius {radius} km", va="center", color=INK)
    ax.set_xlim(0, 12.5)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Distance from the user's anchor (km)")
    ax.set_ylabel("Popularity needed vs a place at 0 km")
    titled(fig, "Far places must be far more popular, less so for explorers",
           "Learned logistic model · travel radius = median distance of the user's past places "
           f"(median user {m['median_radius_km']:.1f} km)")
    save(fig, "L2_learned_distance_curve")

def main():
    """Run this step and write its output (see the module docstring)."""
    for f in (v1_rmse, v5_cold, m1_knn, m3_mf_curve, m4_mf_grid, m5_blend, r2_pareto, t2_timing, t3_gap,
              d1_popularity, l1_learned, l2_distance_curve):
        f()


if __name__ == "__main__":
    main()
