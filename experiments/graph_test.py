from models.bert_attention import (
    BertAttentionExtractor
)
from visualizations.attention_graph import  build_attention_graph

extractor = BertAttentionExtractor()

result = extractor.analyze(
    "The boy kicked the ball because he was angry."
)

matrix = extractor.get_attention_matrix(
    result["outputs"],
    layer=0,
    head=0
)

build_attention_graph(
    result["tokens"],
    matrix
)