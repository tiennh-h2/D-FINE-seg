import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    data = pd.DataFrame(
        {
            "Split": ["Train", "Valid", "Test"],
            "Images": [152, 26, 39],
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
            "grid.linewidth": 0.8,
            "xtick.color": "#475569",
            "ytick.color": "#64748B",
        },
    )

    fig, ax = plt.subplots(figsize=(12, 7.6), dpi=100)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    bars = sns.barplot(
        data=data,
        x="Split",
        y="Images",
        color="#3478E5",
        width=0.58,
        saturation=1,
        ax=ax,
    )

    ax.set_title(
        "Dataset Distribution: Train / Valid / Test",
        loc="left",
        fontsize=24,
        fontweight="bold",
        pad=44,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Images", fontsize=15, fontweight="bold", labelpad=18)
    ax.set_ylim(0, 160)
    ax.set_yticks(range(0, 161, 20))
    ax.tick_params(axis="x", labelsize=16, pad=12)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)

    for patch, value in zip(bars.patches, data["Images"], strict=True):
        patch.set_edgecolor("none")
        patch.set_linewidth(0)
        bars.text(
            patch.get_x() + patch.get_width() / 2,
            value + 3.5,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=19,
            fontweight="bold",
            color="#1E3A5F",
        )

    ax.text(
        1,
        -0.150,
        f"Total: {data['Images'].sum()} images",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=12,
        color="#94A3B8",
    )

    fig.subplots_adjust(left=0.11, right=0.96, top=0.80, bottom=0.145)
    fig.savefig(
        "dataset_distribution.png",
        dpi=200,
        facecolor=fig.get_facecolor(),
        bbox_inches=None,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()