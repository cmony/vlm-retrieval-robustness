from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "paper" / "figures"
NATURAL_CSV = REPO_ROOT / "results" / "natural" / "condition_aggregate.csv"
ROCOCO_CSV = REPO_ROOT / "results" / "rococo" / "rococo_results.csv"
ROCOCO_RELEVANCE_CSV = REPO_ROOT / "results" / "rococo" / "rococo_relevance_crosstab.csv"

MODELS = ["blip_base", "siglip_base", "clip_vit_b16", "clip_vit_b32"]
MODEL_LABEL = {
    "blip_base": "BLIP",
    "siglip_base": "SigLIP",
    "clip_vit_b16": "CLIP ViT-B/16",
    "clip_vit_b32": "CLIP ViT-B/32",
}
MODEL_COLOR = {
    "blip_base": "#1f77b4",
    "siglip_base": "#ff7f0e",
    "clip_vit_b16": "#2ca02c",
    "clip_vit_b32": "#d62728",
}
MODEL_MARKER = {
    "blip_base": "o",
    "siglip_base": "s",
    "clip_vit_b16": "^",
    "clip_vit_b32": "D",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_natural() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(NATURAL_CSV, low_memory=False)
    clean = data[data["family"] == "clean"].set_index(["model", "direction"])["r1"]
    distorted = data[data["family"] != "clean"].copy()
    distorted["delta_r1"] = [
        row.r1 - clean.loc[(row.model, row.direction)]
        for row in distorted.itertuples()
    ]
    return distorted, clean


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def family_means(distorted: pd.DataFrame) -> pd.Series:
    return distorted.groupby(["model", "direction", "family"], sort=False)["delta_r1"].mean()


def make_figure1(distorted: pd.DataFrame) -> None:
    families = ["gaussian_noise", "jpeg", "crop", "brightness", "style"]
    family_labels = ["Noise", "JPEG", "Crop", "Brightness", "Style"]
    grouped = family_means(distorted)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), sharey=True, constrained_layout=True)
    x = np.arange(len(families))
    width = 0.19
    hatches = ["", "//", "..", "xx"]
    for ax, direction, title in zip(
        axes,
        ("image_to_text", "text_to_image"),
        ("Image-to-text", "Text-to-image"),
    ):
        for index, model in enumerate(MODELS):
            values = [grouped.loc[(model, direction, family)] for family in families]
            ax.bar(
                x + (index - 1.5) * width,
                values,
                width=width,
                color=MODEL_COLOR[model],
                hatch=hatches[index],
                edgecolor="white",
                linewidth=0.5,
                label=MODEL_LABEL[model],
                zorder=3,
            )
        ax.set_xticks(x, family_labels)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Mean change in R@1")
        ax.axhline(0, color="0.35", linewidth=0.7)
        ax.grid(axis="y", color="0.88", zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    save_figure(fig, "figure1_revised")


def make_figure2(distorted: pd.DataFrame) -> None:
    families = ["gaussian_noise", "jpeg", "crop", "brightness", "style"]
    family_labels = ["Noise", "JPEG", "Crop", "Brightness", "Style"]
    grouped = family_means(distorted)

    values = []
    for direction in ("image_to_text", "text_to_image"):
        values.append(
            np.array(
                [[grouped.loc[(model, direction, family)] for family in families] for model in MODELS]
            )
        )
    limit = max(abs(np.concatenate([v.ravel() for v in values])).max(), 0.01)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), constrained_layout=True)
    titles = ["Image-to-text", "Text-to-image"]
    image = None
    for ax, matrix, title in zip(axes, values, titles):
        image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(title, pad=10, fontweight="bold")
        ax.set_xticks(range(len(family_labels)), family_labels)
        ax.set_yticks(range(len(MODELS)), [MODEL_LABEL[m] for m in MODELS])
        ax.tick_params(axis="x", rotation=0, pad=5)
        ax.tick_params(axis="y", pad=6)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                text_color = "white" if abs(value) > 0.56 * limit else "black"
                ax.text(col, row, f"{value:+.3f}", ha="center", va="center", color=text_color, fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.025)
    colorbar.set_label("Mean change in R@1")
    fig.suptitle("Family-average retrieval change", fontsize=12, fontweight="bold")
    save_figure(fig, "figure2_revised")


def make_figure3(distorted: pd.DataFrame) -> None:
    specs = [
        ("gaussian_noise", "image_to_text", [0.05, 0.10, 0.20, 0.30], "Image-to-text - Gaussian noise", "Noise sigma"),
        ("gaussian_noise", "text_to_image", [0.05, 0.10, 0.20, 0.30], "Text-to-image - Gaussian noise", "Noise sigma"),
        ("crop", "image_to_text", [0.90, 0.80, 0.70, 0.60], "Image-to-text - Cropping", "Retained crop ratio"),
        ("crop", "text_to_image", [0.90, 0.80, 0.70, 0.60], "Text-to-image - Cropping", "Retained crop ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.2), sharey=True, constrained_layout=True)
    for ax, (family, direction, x_values, title, x_label) in zip(axes.flat, specs):
        subset = distorted[(distorted["family"] == family) & (distorted["direction"] == direction)].copy()
        subset["numeric_value"] = pd.to_numeric(subset["value"], errors="coerce")
        for model in MODELS:
            model_rows = subset[subset["model"] == model].set_index("numeric_value")
            y_values = [model_rows.loc[value, "delta_r1"] for value in x_values]
            ax.plot(
                x_values,
                y_values,
                color=MODEL_COLOR[model],
                marker=MODEL_MARKER[model],
                linewidth=1.8,
                markersize=4.5,
                label=MODEL_LABEL[model],
            )
        ax.axhline(0, color="0.35", linewidth=0.7)
        ax.set_title(title, pad=7, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Change in R@1")
        ax.set_xticks(x_values)
        ax.grid(True, color="0.88", linewidth=0.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.035))
    save_figure(fig, "figure3_revised")


def make_figure4(distorted: pd.DataFrame) -> None:
    operations = ["grayscale", "invert"]
    operation_labels = ["Grayscale", "Inversion"]
    colors = ["#6baed6", "#cb181d"]
    subset = distorted[distorted["family"] == "style"].copy()
    subset["operation"] = subset["value"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), sharey=True, constrained_layout=True)
    x = np.arange(len(MODELS))
    width = 0.34
    for ax, direction, title in zip(
        axes,
        ("image_to_text", "text_to_image"),
        ("Image-to-text", "Text-to-image"),
    ):
        direction_rows = subset[subset["direction"] == direction]
        for index, (operation, label, color) in enumerate(zip(operations, operation_labels, colors)):
            rows = direction_rows[direction_rows["operation"] == operation].set_index("model")
            values = [rows.loc[model, "delta_r1"] for model in MODELS]
            ax.bar(
                x + (index - 0.5) * width,
                values,
                width=width,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=label,
                zorder=3,
            )
        ax.set_xticks(x, [MODEL_LABEL[model] for model in MODELS])
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Change in R@1")
        ax.axhline(0, color="0.35", linewidth=0.7)
        ax.grid(axis="y", color="0.88", zorder=0)
        ax.legend(frameon=False, loc="lower left")
    save_figure(fig, "figure4_revised")


def make_figure5() -> None:
    """Exact-item (light) and source-aware (solid, hatched) change in R@1.

    Exact-item values come from results/rococo/rococo_results.csv; source-aware
    values come from results/rococo/rococo_relevance_crosstab.csv, written by
    scripts/revision/rococo_relevance.py.
    """
    from matplotlib.patches import Patch

    exact = pd.read_csv(ROCOCO_CSV)
    relevance = pd.read_csv(ROCOCO_RELEVANCE_CSV)
    relevance = relevance[relevance["queries"] == "all"]
    label_to_model = {label: model for model, label in MODEL_LABEL.items()}
    relevance = relevance.assign(model=relevance["model"].map(label_to_model))
    source_aware = relevance.set_index(["model", "perturbation"])["delta_source_aware"]
    exact = exact.assign(perturbation=exact["perturbation"].replace({"mix_0.9": "image_mixing"}))
    exact_delta = exact.set_index(["model", "perturbation"])["delta_r1"]

    perturbations = ["same_concept", "diff_concept", "rand_voca", "danger"]
    perturbation_labels = ["Same concept", "Different concept", "Random vocabulary", "Danger"]
    hatches = ["", "//", "..", "xx"]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1.55, 1]}, constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(perturbations))
    width = 0.19
    for index, model in enumerate(MODELS):
        positions = x + (index - 1.5) * width
        ax.bar(positions, [exact_delta[(model, p)] for p in perturbations], width=width,
               color=MODEL_COLOR[model], alpha=0.30, edgecolor="none", zorder=2)
        ax.bar(positions, [source_aware[(model, p)] for p in perturbations], width=width,
               color=MODEL_COLOR[model], hatch=hatches[index], edgecolor="white", linewidth=0.5,
               label=MODEL_LABEL[model], zorder=3)
    ax.set_xticks(x, perturbation_labels, rotation=12, ha="right")
    ax.set_ylabel("Change in R@1 (image-to-text)")
    ax.set_title("(a) Caption injection", fontweight="bold")
    ax.axhline(0, color="0.35", linewidth=0.7)
    ax.grid(axis="y", color="0.88", zorder=0)
    handles, _ = ax.get_legend_handles_labels()
    handles += [Patch(facecolor="0.45", alpha=0.30, label="Exact-item (light)"),
                Patch(facecolor="0.45", label="Source-aware (solid)")]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower left")

    ax = axes[1]
    x2 = np.arange(len(MODELS))
    colors = [MODEL_COLOR[m] for m in MODELS]
    sa = [source_aware[(m, "image_mixing")] for m in MODELS]
    ax.bar(x2, [exact_delta[(m, "image_mixing")] for m in MODELS], 0.6, color=colors, alpha=0.30,
           edgecolor="none", zorder=2)
    ax.bar(x2, sa, 0.6, color=colors, hatch=hatches, edgecolor="white", linewidth=0.5, zorder=3)
    for xi, value in zip(x2, sa):
        offset = 0.008 if value >= 0 else -0.008
        ax.text(xi, value + offset, f"{value:+.3f}", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=7.5)
    ax.set_xticks(x2, [MODEL_LABEL[m] for m in MODELS], rotation=18, ha="right")
    ax.set_ylabel("Change in R@1 (text-to-image)")
    ax.set_title("(b) Image mixing", fontweight="bold")
    ax.axhline(0, color="0.35", linewidth=0.7)
    ax.grid(axis="y", color="0.88", zorder=0)
    ax.set_ylim(-0.32, 0.03)
    save_figure(fig, "figure5_revised")


def main() -> None:
    configure_style()
    distorted, _ = load_natural()
    make_figure1(distorted)
    make_figure2(distorted)
    make_figure3(distorted)
    make_figure4(distorted)
    make_figure5()
    print("Wrote revised Figures 1-5 to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
