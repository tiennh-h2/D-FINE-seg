import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    data = pd.DataFrame(
        {
            "Model": [
                "SAM2-UNeXT",
                "SAM2-UNeXT",
                "SAM2-SAM3-UNeXT",
                "SAM2-SAM3-UNeXT",
                "DFINE-SEG",
                "DFINE-SEG",
            ],
            "Metric": ["mDice", "mIoU", "mDice", "mIoU", "mDice", "mIoU"],
            "Score": [0.913, 0.858, 0.933, 0.884, 0.921, 0.861],
        }
    )

    sns.set_theme(
        style="whitegrid",
        context="talk",
        font="DejaVu Sans",
        rc={
            "axes.edgecolor": "#7B8491",
            "axes.labelcolor": "#475569",
            "axes.titlecolor": "#173F83",
            "grid.color": "#D9DEE7",
            "grid.linewidth": 0.9,
            "xtick.color": "#475569",
            "ytick.color": "#64748B",
        },
    )

    fig, ax = plt.subplots(figsize=(15, 9), dpi=100)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    chart = sns.barplot(
        data=data,
        x="Model",
        y="Score",
        hue="Metric",
        hue_order=["mDice", "mIoU"],
        palette={"mDice": "#3478E5", "mIoU": "#82B936"},
        saturation=1,
        width=0.66,
        ax=ax,
    )

    ax.set_title(
        "Wall Segmentation Performance Comparison",
        loc="left",
        fontsize=25,
        fontweight="bold",
        pad=50,
    )
    ax.text(
        0,
        1.035,
        "Dataset: fbm_wall_seg_20260725 (39 images)",
        transform=ax.transAxes,
        fontsize=15,
        color="#64748B",
        ha="left",
    )

    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=16, fontweight="bold", labelpad=18)
    ax.set_ylim(0.80, 0.96)
    ax.set_yticks([0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96])
    ax.tick_params(axis="x", labelsize=15, pad=12)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)

    legend = ax.legend(
        title=None,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#FFFFFF",
        fontsize=14,
    )
    legend.get_frame().set_alpha(0.96)

    for container in chart.containers:
        ax.bar_label(
            container,
            fmt="%.3f",
            padding=6,
            fontsize=14,
            fontweight="bold",
            color="#1E3A5F",
        )

    fig.subplots_adjust(left=0.10, right=0.96, top=0.79, bottom=0.15)
    fig.savefig(
        "wall_segmentation_performance.png",
        dpi=100,
        facecolor=fig.get_facecolor(),
        bbox_inches=None,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()