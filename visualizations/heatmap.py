import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_heatmap(
        matrix,
        tokens,
        title="Attention Heatmap"
    ):

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis"
    )

    plt.title(title)

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        "heatmap.png",
        bbox_inches="tight"
    )

    plt.show()