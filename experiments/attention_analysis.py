from models.bert_attention import (
    BertAttentionExtractor
)

from visualizations.attention_analysis import  describe_token_attention, analyze_all_tokens

extractor = BertAttentionExtractor()

result = extractor.analyze(
    "The boy kicked the ball because he was angry."
)

matrix = extractor.get_attention_matrix(
    result["outputs"],
    layer=0,
    head=0
)

tokens = result["tokens"]

for i, token in enumerate(tokens):
    print(i, token)

he_index = 7
describe_token_attention(
    matrix,
    tokens,
    he_index
)

analyze_all_tokens(
    matrix,
    tokens,
)