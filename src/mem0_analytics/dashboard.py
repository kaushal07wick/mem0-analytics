import os, time, sqlite3, pandas as pd, numpy as np
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

console = Console()
DB_PATH = os.path.expanduser("~/.mem0_metrics.db")
REFRESH_SEC = 60


def get_conn():
    return sqlite3.connect(DB_PATH)


def safe_query(q):
    try:
        with get_conn() as conn:
            df = pd.read_sql(q, conn)
            return df.replace([np.inf, -np.inf], np.nan).fillna(0)
    except Exception:
        return pd.DataFrame()


def chart_bar(df, x, y, title, color="cyan", unit=""):
    if df.empty or y not in df.columns:
        return Panel("No data", title=title, border_style="dim", padding=(0, 1))

    df = df[df[y] > 0].sort_values(y, ascending=False).head(6)
    if df.empty:
        return Panel("No numeric data", title=title, border_style="dim", padding=(0, 1))

    max_val = df[y].max() or 1
    bars = [
        f"[{color}]{str(row[x])[:22]:<22}[/] ▏{'█' * max(1, int((row[y] / max_val) * 20))} {row[y]:.2f}{unit}"
        for _, row in df.iterrows()
    ]
    return Panel("\n".join(bars), title=title, border_style=color, padding=(0, 1))


def mini_chart(series, width=40):
    if len(series) < 2:
        return "─" * width
    vals = (series - series.min()) / (series.max() - series.min() + 1e-9)
    chars = " ▁▂▃▄▅▆▇█"
    idx = (vals * (len(chars) - 1)).astype(int)
    return "".join(chars[i] for i in idx[-width:])


# === KPI Renderers ===
def render_latency_by_vectorstore(df):
    g = df.groupby("provider_vectorstore")["avg_latency_ms"].mean().reset_index()
    return chart_bar(g, "provider_vectorstore", "avg_latency_ms", "Average Response Latency (ms) grouped by Vector Store", "cyan", " ms")


def render_latency_per_model(df):
    g = df.groupby("model_llm")["avg_latency_ms"].mean().reset_index()
    return chart_bar(g, "model_llm", "avg_latency_ms", "Mean Latency per Model (ms)", "bright_blue", " ms")


def render_latency_by_operation(df):
    g = df.groupby("function_name")["avg_latency_ms"].mean().reset_index()
    return chart_bar(g, "function_name", "avg_latency_ms", "Latency by Operations in Mem0", "magenta", " ms")


def render_ttfr(df):
    g = df.groupby("provider_vectorstore")["avg_vector_latency"].mean().reset_index()
    return chart_bar(g, "provider_vectorstore", "avg_vector_latency", "Average TTFR (ms) by Vector Store", "green", " ms")


def render_embedder(df):
    g = df.groupby("provider_embedder")["avg_embed_latency"].mean().reset_index()
    return chart_bar(g, "provider_embedder", "avg_embed_latency", "Embedder Latency (ms)", "cyan", " ms")


def render_p95(df):
    g = df.groupby("function_name")["latency_p95"].mean().reset_index()
    return chart_bar(g, "function_name", "latency_p95", "P95 Latency (ms)", "bright_yellow", " ms")


def render_cache(df):
    g = df.groupby("function_name")["cache_effectiveness"].mean().reset_index()
    return chart_bar(g, "function_name", "cache_effectiveness", "Cache Effectiveness", "yellow", " %")


def render_trend(df):
    metrics = [
        ("avg_latency_ms", "cyan"),
        ("avg_vector_latency", "green"),
        ("avg_embed_latency", "magenta"),
    ]
    lines = []
    for metric, color in metrics:
        d = df.groupby("ts")[metric].mean().reset_index().sort_values("ts")
        if d.empty or np.allclose(d[metric], 0):
            continue
        if np.allclose(d[metric], d[metric].iloc[0]):
            lines.append(f"[{color}]{metric:<20}[/] ▔ flatline (no change)")
            continue
        series = d[metric].tail(40)
        line = mini_chart(series)
        lines.append(f"[{color}]{metric.replace('_', ' ')[:18]:<18}[/] {line}  {series.max():.1f}ms")
    if not lines:
        return Panel("No dynamic trends detected", title="Performance Trends", border_style="dim")
    return Panel("\n".join(lines), title="Performance Trends (Last 40 samples)", border_style="green", padding=(0, 1))


def render_status(df):
    if df.empty:
        return Panel("🕓 Idle", title="System Status", border_style="yellow", padding=(0, 1))
    err = float(df["error_rate"].mean())
    succ = float(df["success_rate"].mean() * 100)
    if err > 0.3:
        txt = f"⚠ Problem ({succ:.1f}% success)"
        style = "red"
    elif err > 0.05:
        txt = f"🟡 Degraded ({succ:.1f}% success)"
        style = "yellow"
    else:
        txt = f"✅ Stable ({succ:.1f}% success)"
        style = "green"
    return Panel(txt, title="System Status", border_style=style, padding=(0, 1))


def get_data():
    return safe_query("SELECT * FROM mem0_kpi ORDER BY ts DESC;")


def run_dashboard():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="row1", ratio=4),
        Layout(name="row2", ratio=4),
        Layout(name="row3", ratio=4),
        Layout(name="footer", size=2),
    )

    layout["row1"].split_row(
        Layout(name="lat_vec"),
        Layout(name="lat_model"),
        Layout(name="lat_op"),
    )

    layout["row2"].split_row(
        Layout(name="ttfr"),
        Layout(name="embedder"),
        Layout(name="p95"),
    )

    layout["row3"].split_row(
        Layout(name="cache"),
        Layout(name="trend"),
        Layout(name="status", size=20),
    )

    layout["header"].update(
        Panel(Text("📊 MEM0 KPI DASHBOARD", justify="center", style="bold cyan"), border_style="cyan")
    )

    with Live(layout, refresh_per_second=1, screen=False):
        while True:
            try:
                df = get_data()
                if df.empty:
                    for sec in (
                        "lat_vec", "lat_model", "lat_op", "ttfr",
                        "embedder", "p95", "cache", "trend", "status"
                    ):
                        layout[sec].update(Panel("Waiting for data...", border_style="yellow"))
                else:
                    layout["lat_vec"].update(render_latency_by_vectorstore(df))
                    layout["lat_model"].update(render_latency_per_model(df))
                    layout["lat_op"].update(render_latency_by_operation(df))
                    layout["ttfr"].update(render_ttfr(df))
                    layout["embedder"].update(render_embedder(df))
                    layout["p95"].update(render_p95(df))
                    layout["cache"].update(render_cache(df))
                    layout["trend"].update(render_trend(df))
                    layout["status"].update(render_status(df))

                layout["footer"].update(
                    Panel(f"Last update: {datetime.now():%H:%M:%S}", border_style="dim", padding=(0, 1))
                )
                time.sleep(REFRESH_SEC)
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Dashboard error: {e}[/red]")
                time.sleep(REFRESH_SEC)


if __name__ == "__main__":
    run_dashboard()
