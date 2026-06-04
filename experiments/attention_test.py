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