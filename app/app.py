import streamlit as st
from models.bert_attention import (
    BertAttentionExtractor
)
from visualizations.attention_analysis import (
    get_top_attention_targets
)
from visualizations.streamlit_heatmap import (
    create_attention_heatmap
)

@st.cache_resource
def load_extractor():
    return BertAttentionExtractor()

extractor = load_extractor()

st.set_page_config(
    page_title="BERT Attention Explorer",
    page_icon="🧠",
    layout="wide"
)

st.sidebar.title("Attention Controls")

selected_layer = st.sidebar.selectbox(
    "Select Layer",
    options=list(range(1, 13)),
    index=0
)

selected_head = st.sidebar.selectbox(
    "Select Head",
    options=list(range(1, 13)),
    index=0
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"Layer: {selected_layer}"
)

st.sidebar.write(
    f"Head: {selected_head}"
)

st.title(
    "BERT Attention Explorer"
)

st.write(
    """
    Explore attention patterns
    inside BERT.
    """
)

left_col, right_col = st.columns([2, 1])

with left_col:

    st.subheader("Input Text")

    text = st.text_area(
        "Enter text to analyze",
        value="The boy kicked the ball because he was angry.",
        height=150
    )

    analyze_button = st.button(
        "Analyze",
        type="primary"
    )

layer_idx = selected_layer - 1
head_idx = selected_head - 1

if analyze_button:

    with st.spinner("Running BERT analysis..."):

        result = extractor.analyze(text)
    
    st.session_state["result"] = result
    st.session_state["text"] = text

if "result" in st.session_state:

    result = st.session_state["result"]

    tokens = result["tokens"]

    matrix = extractor.get_attention_matrix(
    result["outputs"],
    layer=layer_idx,
    head=head_idx
    )

    heatmap_fig = create_attention_heatmap(
        matrix,
        tokens,
        title=f"Layer {selected_layer} | Head {selected_head}"
    )

    st.success(
        "Analysis Complete"
    )

    with right_col:

        st.subheader("Statistics")

        st.metric(
            "Token Count",
            len(tokens)
        )

        st.metric(
            "Transformer Layers",
            len(result["outputs"].attentions)
        )

        st.metric(
            "Selected Layer",
            selected_layer
        )

        st.metric(
            "Selected Head",
            selected_head
        )

    st.divider()

    st.subheader("Tokenized Output")

    st.write(" | ".join(tokens))

    st.subheader("Token Indices")

    token_data = []

    for i, token in enumerate(tokens):

        token_data.append(
            {
                "Index": i,
                "Token": token
            }
        )

    st.table(token_data)

    st.divider()

    selected_token = st.selectbox(
        "Choose Token",
        options=list(range(len(tokens))),
        format_func=lambda x:
            f"{x}: {tokens[x]}"
    )

    st.write(
        f"Selected Token: **{tokens[selected_token]}**"
    )

    top_targets = get_top_attention_targets(
        matrix,
        tokens,
        selected_token,
        top_k=5
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Heatmap",
            "Matrix",
            "Token Analysis"
        ]
    )

    with tab1:

        st.subheader(
            "Attention Heatmap"
        )

        st.info(
            f"Viewing Layer {selected_layer}, Head {selected_head}"
        )

        st.pyplot(
            heatmap_fig
        )

    with tab2:

        st.subheader(
            "Attention Matrix"
        )

        st.write(
            "Shape:",
            matrix.shape
        )

        st.dataframe(
            matrix
        )

    with tab3:

        st.subheader(
            "Top Attention Targets"
        )

        for token, score in top_targets:

            st.write(
                f"{token}: {float(score):.4f}"
            )
    
    with st.expander(
        "Show Raw Token List"
    ):

        st.write(
            tokens
        )

    with st.expander(
        "Show Raw Matrix"
    ):

        st.dataframe(
            matrix
        )

else:

    st.info(
        "Enter text and click Analyze to inspect BERT tokenization."
    )