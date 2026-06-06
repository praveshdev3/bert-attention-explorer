import matplotlib.pyplot as plt
import seaborn as sns


def create_attention_heatmap(
        matrix,
        tokens,
        title="Attention Heatmap"
):

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    sns.heatmap(
        matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        ax=ax
    )

    ax.set_title(title)

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()

    return fig