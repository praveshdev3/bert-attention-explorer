import networkx as nx
import matplotlib.pyplot as plt

def build_attention_graph(
        tokens,
        matrix
):
    G = nx.DiGraph()

    special_tokens = {"[CLS]", "[SEP]"}

    valid_indices = [
        i for i, token in enumerate(tokens)
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
        weight=float(row[target_index])
    )
        
    pos = nx.spring_layout(G,seed=42)
    plt.figure(figsize=(12,8))
    nx.draw_networkx_nodes(
    G,
    pos
)
    nx.draw_networkx_labels(
    G,
    pos
)
    nx.draw_networkx_edges(
    G,
    pos,
    arrows=True
)
    plt.title(
    "Attention Graph"
)

    plt.axis("off")

    plt.show()