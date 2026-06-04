def get_token_attention(
        matrix,
        tokens,
        token_index
):
    row = matrix[token_index]

    results = []

    for token, score in zip(tokens, row):
        results.append(
            (
                token,
                float(score)
            )
        )

    return results

def get_top_attention_targets(
        matrix,
        tokens,
        token_index,
        top_k=5
):
    row = matrix[token_index]

    pairs = list(
        zip(tokens, row)
    )

    pairs = [
    (token, score)
    for idx, (token, score)
    in enumerate(zip(tokens, row))
    if idx != token_index
    ]

    pairs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return pairs[:top_k]

def describe_token_attention(
        matrix,
        tokens,
        token_index,
        top_k=3
):
    source_token = tokens[token_index]

    top = get_top_attention_targets(
        matrix,
        tokens,
        token_index,
        top_k
    )

    print(
        f"\nToken: {source_token}"
    )

    print(
        "Top Attention Targets:"
    )

    for token, score in top:
        print(
            f"{token}: {score:.3f}"
        )

def analyze_all_tokens(
        matrix,
        tokens
):
    for idx, token in enumerate(tokens):
        top = get_top_attention_targets(
        matrix,
        tokens,
        idx,
        top_k=1
    )
        
        print(
        token,
        "→",
        top[0][0]
    )   