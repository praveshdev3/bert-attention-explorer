import math

import matplotlib.pyplot as plt
import seaborn as sns


def create_multi_head_heatmap(
        outputs,
        tokens,
        layer_idx,
        num_heads=4
):

    attentions = outputs.attentions

    layer_attention = attentions[layer_idx]

    cols = min(4, num_heads)

    rows = math.ceil(num_heads / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows)
    )

    if num_heads == 1:

        axes = [axes]

    else:

        axes = axes.flatten()

    for head_idx in range(num_heads):

        matrix = (
            layer_attention[0][head_idx]
            .detach()
            .cpu()
            .numpy()
        )

        sns.heatmap(
            matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            ax=axes[head_idx],
            cbar=False
        )

        axes[head_idx].set_title(
            f"Head {head_idx + 1}"
        )

        axes[head_idx].tick_params(
            axis="x",
            rotation=45
        )

        axes[head_idx].tick_params(
            axis="y",
            rotation=0
        )

    for i in range(
            num_heads,
            len(axes)
    ):

        axes[i].axis("off")

    plt.tight_layout()

    return fig