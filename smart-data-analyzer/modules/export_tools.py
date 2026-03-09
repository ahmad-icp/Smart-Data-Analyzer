import io
import pandas as pd


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def statistics_to_csv(stats_df: pd.DataFrame) -> bytes:
    """Convert a statistics DataFrame to CSV bytes for download."""
    return stats_df.to_csv(index=True).encode("utf-8")


def plot_to_image_bytes(fig, width: int = 800, height: int = 600) -> bytes:
    """Render a Plotly figure to PNG bytes. Requires plotly[kaleido]."""
    try:
        import plotly.io as pio

        return pio.to_image(fig, format="png", width=width, height=height)
    except Exception as e:
        raise RuntimeError(
            "Failed to render plot to image. Ensure plotly[kaleido] is installed." + str(e)
        )
