from httpx import head
from transformers import BertTokenizer, BertModel

class BertAttentionExtractor:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        self.model = BertModel.from_pretrained(
            "bert-base-uncased",
            output_attentions=True
        )

    def process_text(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        outputs = self.model(**inputs)

        return inputs, outputs

    def get_tokens(self, inputs):

        return self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0]
        )

    def get_attention_matrix(
            self,
            outputs,
            layer,
            head
        ):

        num_layers = len(outputs.attentions)

        num_heads = outputs.attentions[0].shape[1]

        if layer < 0 or layer >= num_layers:
            raise ValueError(
            f"Layer must be between 0 and {num_layers - 1}"
        )

        if head < 0 or head >= num_heads:
            raise ValueError(
            f"Head must be between 0 and {num_heads - 1}"
        )
    
        return (
            outputs.attentions[layer]
            [0, head]
            .detach()
            .numpy()
        )

    def analyze(self, text):

        inputs, outputs = self.process_text(text)

        tokens = self.get_tokens(inputs)

        return {
            "tokens": tokens,
            "outputs": outputs
        }
    
    def get_top_connections(
        self,
        matrix,
        tokens,
        token_index,
        top_k=5
    ):
        
        row = matrix[token_index]

        pairs = list(zip(tokens, row))

        pairs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return pairs[:top_k]