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
from visualizations.streamlit_graph import (
    create_attention_graph
)
from visualizations.multi_head import (
    create_multi_head_heatmap
)
from utils.export import (
    figure_to_png_bytes,
    matrix_to_csv_bytes
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

show_edge_labels = st.sidebar.checkbox(
    "Show Edge Weights",
    value=True
)

num_compare_heads = st.sidebar.slider(
    "Heads To Compare",
    min_value=1,
    max_value=12,
    value=4
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"Layer: {selected_layer}"
)

st.sidebar.write(
    f"Head: {selected_head}"
)

st.title("BERT Attention Explorer")

st.markdown(
    """
    Explore how BERT distributes attention across tokens.

    Features:
    - Attention Heatmaps
    - Token-Level Analysis
    - Attention Graphs
    - Multi-Head Comparison
    - Export Visualizations
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

    if not text.strip():

        st.warning(
            "Please enter some text."
        )

        st.stop()

    if len(text.split()) > 100:

        st.warning(
            "Please keep input under 100 words."
        )

        st.stop()

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

    graph_fig = create_attention_graph(
        tokens,
        matrix,
        show_edge_labels
    )

    multi_head_fig = create_multi_head_heatmap(
        result["outputs"],
        tokens,
        layer_idx,
        num_heads=num_compare_heads
    )

    heatmap_png = figure_to_png_bytes(
        heatmap_fig
    )

    graph_png = figure_to_png_bytes(
        graph_fig
    )

    matrix_csv = matrix_to_csv_bytes(
        matrix,
        tokens
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Heatmap",
            "Matrix",
            "Token Analysis",
            "Graph",
            "Multi-Head Compare"
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

        st.download_button(
            label="Download Heatmap PNG",
            data=heatmap_png,
            file_name=(
                f"heatmap_layer_"
                f"{selected_layer}_"
                f"head_{selected_head}.png"
            ),
            mime="image/png"
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

        st.download_button(
            label="Download Matrix CSV",
            data=matrix_csv,
            file_name=(
                f"matrix_layer_"
                f"{selected_layer}_"
                f"head_{selected_head}.csv"
            ),
            mime="text/csv"
        )

    with tab3:

        st.subheader(
            "Top Attention Targets"
        )

        for token, score in top_targets:

            st.write(
                f"{token}: {float(score):.4f}"
            )

    with tab4:

        st.subheader(
            "Attention Graph"
        )

        st.info(
            """
            Each token points to the token
            receiving its strongest attention.

            Thicker arrows indicate
            stronger attention weights.
            """
        )

        st.pyplot(
            graph_fig
        )

        st.download_button(
            label="Download Graph PNG",
            data=graph_png,
            file_name=(
                f"graph_layer_"
                f"{selected_layer}_"
                f"head_{selected_head}.png"
            ),
            mime="image/png"
        )
    
    with tab5:

        st.subheader(
            "Multi-Head Comparison"
        )

        st.info(
            f"""
            Comparing first {num_compare_heads} heads
            of Layer {selected_layer}
            """
        )

        st.pyplot(
            multi_head_fig
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

st.markdown("---")

st.caption(
    """
    Built using Streamlit, Hugging Face Transformers,
    Matplotlib, NetworkX and BERT.
    """
)