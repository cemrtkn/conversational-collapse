import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # Note: CSV should be read with read_csv, not read_parquet
    results = pd.read_csv("results/drift_experiment_20260127_081134.csv")
    avg_entropies = pd.read_parquet("avg_entropies2.parquet")

    # Get texts and entropy values
    texts = results["content"].tolist()
    entropies = avg_entropies["avg_entropy"].tolist()

    # Filter out NaN entries
    valid_data = [
        (text, ent) for text, ent in zip(texts, entropies) if not np.isnan(ent)
    ]
    texts, entropies = zip(*valid_data) if valid_data else ([], [])

    # Truncate text for display (first 80 chars)
    display_texts = [t[:80] + "..." if len(t) > 80 else t for t in texts]

    # Normalize entropies for colormap
    # Lower entropy = red (bad), higher entropy = green (good)
    ent_array = np.array(entropies)
    ent_min, ent_max = ent_array.min(), ent_array.max()

    if ent_max > ent_min:
        normalized = (ent_array - ent_min) / (ent_max - ent_min)
    else:
        normalized = np.ones_like(ent_array) * 0.5

    # Create colormap: red (low entropy/bad)
    # -> yellow -> green (high entropy/good)
    cmap = plt.cm.RdYlGn

    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(texts) * 0.4)))

    # Plot each text entry with its color
    y_positions = np.arange(len(display_texts))[
        ::-1
    ]  # Reverse for top-to-bottom

    for i, (text, norm_val, raw_ent) in enumerate(
        zip(display_texts, normalized, entropies)
    ):
        color = cmap(norm_val)
        # Add background rectangle
        rect = plt.Rectangle(
            (-0.02, y_positions[i] - 0.4),
            1.04,
            0.8,
            facecolor=color,
            alpha=0.7,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )
        ax.add_patch(rect)

        # Add text
        ax.text(
            0.01,
            y_positions[i],
            f"[{i}] (ent={raw_ent:.2f}) {text}",
            va="center",
            ha="left",
            fontsize=8,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )

    # Configure axes
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(texts) - 0.5)
    ax.axis("off")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(ent_min, ent_max))
    sm.set_array([])
    cbar = plt.colorbar(
        sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.02
    )
    cbar.set_label(
        "Average Entropy (red=low/worse, green=high/better)", fontsize=10
    )

    plt.title(
        "Text Entries Colored by Average Entropy",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        "text_entropy_visualization2.png", dpi=150, bbox_inches="tight"
    )
    plt.show()
    print("Saved to text_entropy_visualization2.png")


if __name__ == "__main__":
    main()
