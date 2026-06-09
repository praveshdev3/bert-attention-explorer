import io
import pandas as pd


def matrix_to_csv_bytes(
        matrix,
        tokens
):

    df = pd.DataFrame(
        matrix,
        index=tokens,
        columns=tokens
    )

    return df.to_csv().encode(
        "utf-8"
    )

def figure_to_png_bytes(fig):

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight"
    )

    buffer.seek(0)

    return buffer.getvalue()