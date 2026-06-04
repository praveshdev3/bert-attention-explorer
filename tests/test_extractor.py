from models.bert_attention import BertAttentionExtractor
extractor = BertAttentionExtractor()

result = extractor.analyze(
    "Transformers changed NLP."
)

assert len(result["tokens"]) > 0