from models.bert_attention import (
    BertAttentionExtractor
)

extractor = BertAttentionExtractor()

result = extractor.analyze(
    "The boy kicked the ball because he was angry."
)

print(result["tokens"])

matrix = extractor.get_attention_matrix(
    result["outputs"],
    layer=0,
    head=0
)

print(matrix.shape)

