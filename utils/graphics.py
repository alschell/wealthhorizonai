import matplotlib.pyplot as plt
import io
import base64
import logging

logger = logging.getLogger(__name__)

async def generate_performance_graph(series_dict):
    fig, ax = plt.subplots(figsize=(12, 8))
    for label, series in series_dict.items():
        ax.plot(series, label=label)
    ax.legend()
    ax.set_title("Performance Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{img_base64}"
