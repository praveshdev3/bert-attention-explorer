from models.bert_attention import (
    BertAttentionExtractor
)

from visualizations.heatmap import (
    plot_attention_heatmap
)

extractor = BertAttentionExtractor()

result = extractor.analyze(
    "The boy kicked the ball because he was angry."
)

for head in range(4):

    matrix = extractor.get_attention_matrix(
        result["outputs"],
        layer=0,
        head=head
    )

    plot_attention_heatmap(
        matrix,
        result["tokens"],
        title=f"Layer 1 Head {head+1}"
    )