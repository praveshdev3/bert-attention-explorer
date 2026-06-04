from transformers import BertTokenizer, BertModel

print("Loading tokenizer...")

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

print("Loading model...")

model = BertModel.from_pretrained(
    "bert-base-uncased",
    output_attentions=True
)

text = "The boy kicked the ball because he was angry."

inputs = tokenizer(
    text,
    return_tensors="pt"
)

outputs = model(**inputs)

print("\nSuccess!")
print(
    "Number of layers:",
    len(outputs.attentions)
)
layer0 = outputs.attentions[0]

print(type(layer0))
print(layer0.shape)
tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print(tokens)
print("Number of tokens:")
print(len(tokens))

head0 = layer0[0, 0]
print(head0.shape)
print(head0)
matrix = head0.detach().numpy()
print(matrix)

for token, score in zip(tokens, matrix[7]):
    print(token, round(float(score), 3))

def top_attention(tokens, row, top_k=5):
    pairs = list(zip(tokens, row))

    pairs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return pairs[:top_k]

top = top_attention(
    tokens,
    matrix[7]
)

print(top)
print(matrix[7].sum())