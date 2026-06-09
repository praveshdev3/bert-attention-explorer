import networkx as nx
import matplotlib.pyplot as plt


def create_attention_graph(
        tokens,
        matrix,
        show_edge_labels=True
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

        node_name = f"{i}:{tokens[i]}"

        G.add_node(node_name)

    for i in valid_indices:

        source_token = f"{i}:{tokens[i]}"

        row = matrix[i].copy()

        for idx, token in enumerate(tokens):

            if token in special_tokens:
                row[idx] = float("-inf")

        row[i] = float("-inf")

        target_index = row.argmax()

        target_token = f"{target_index}:{tokens[target_index]}"

        attention_weight = float(
            row[target_index]
        )

        G.add_edge(
            source_token,
            target_token,
            weight=attention_weight
        )

    pos = nx.spring_layout(
        G,
        seed=42
    )

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=9
    )

    weights = [
        G[u][v]["weight"]
        for u, v in G.edges()
    ]

    edge_widths = [
        1 + weight * 8
        for weight in weights
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        arrows=True,
        ax=ax
    )

    if show_edge_labels:

        edge_labels = {
            (u, v): f"{d['weight']:.2f}"
            for u, v, d
            in G.edges(data=True)
        }

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=8
        )

    ax.set_title(
        "Attention Graph"
    )

    ax.axis("off")

    plt.tight_layout()

    return fig