import networkx as nx
import matplotlib.pyplot as plt


def create_attention_graph(
        tokens,
        matrix
):

    G = nx.DiGraph()

    special_tokens = {
        "[CLS]",
        "[SEP]"
    }

    valid_indices = [
        i
        for i, token in enumerate(tokens)
        if token not in special_tokens
    ]

    for i in valid_indices:

        G.add_node(tokens[i])

    for i in valid_indices:

        source_token = tokens[i]

        row = matrix[i].copy()

        for idx, token in enumerate(tokens):

            if token in special_tokens:
                row[idx] = -1

        row[i] = -1

        target_index = row.argmax()

        target_token = tokens[target_index]

        G.add_edge(
            source_token,
            target_token,
            weight=float(
                row[target_index]
            )
        )

    pos = nx.spring_layout(
        G,
        seed=42
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        ax=ax
    )

    ax.set_title(
        "Attention Graph"
    )

    ax.axis("off")

    return fig