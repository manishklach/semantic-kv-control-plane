"""Static HTML dashboard generation for demo runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_static_dashboard(
    results_csv: Path,
    plots_dir: Path,
    output_path: Path,
    title: str = "Semantic KV Control Plane Dashboard",
) -> Path:
    """Create a dependency-free static dashboard from CSV and PNG artifacts."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(results_csv) if results_csv.exists() else pd.DataFrame()
    cards = _metric_cards(results)
    summary = _executive_summary(results)
    rows = results.to_html(index=False, classes="results", border=0) if not results.empty else ""
    images_by_tab = _images_by_tab(plots_dir, output_path.parent)
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      margin:0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      background:#090d18;
      color:#e5e7eb;
    }}
    header {{
      padding:40px 48px 24px;
      border-bottom:1px solid #1f2937;
      background:linear-gradient(180deg,#111827,#090d18);
    }}
    h1 {{ margin:0 0 8px; font-size:34px; letter-spacing:0; }}
    p {{ color:#94a3b8; max-width:880px; line-height:1.6; }}
    main {{ padding:28px 48px 56px; }}
    .cards {{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
      gap:14px;
      margin-bottom:28px;
    }}
    .card {{ border:1px solid #1f2937; background:#111827; border-radius:8px; padding:16px; }}
    .label {{ color:#94a3b8; font-size:13px; }}
    .value {{ font-size:24px; margin-top:8px; font-weight:700; }}
    .grid {{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(420px,1fr));
      gap:18px;
      align-items:start;
    }}
    .tabs {{ display:flex; gap:8px; flex-wrap:wrap; margin:20px 0; }}
    .tab {{
      color:#cbd5e1;
      border:1px solid #334155;
      padding:8px 12px;
      border-radius:8px;
      background:#111827;
      font-size:13px;
    }}
    .panel {{ margin-top:28px; }}
    .panel h2 {{ font-size:22px; margin:0 0 8px; }}
    .summary {{
      border:1px solid #334155;
      background:#0f172a;
      border-radius:8px;
      padding:18px;
      margin-bottom:24px;
    }}
    figure {{
      margin:0;
      border:1px solid #1f2937;
      background:#0f172a;
      border-radius:8px;
      overflow:hidden;
    }}
    img {{ width:100%; display:block; }}
    figcaption {{ padding:12px 14px; color:#cbd5e1; border-top:1px solid #1f2937; }}
    table.results {{ margin-top:30px; width:100%; border-collapse:collapse; font-size:13px; }}
    table.results th, table.results td {{
      padding:9px 10px;
      border-bottom:1px solid #1f2937;
      text-align:left;
    }}
    table.results th {{ color:#93c5fd; background:#111827; position:sticky; top:0; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>
      Simulation-only observability view for tier occupancy, prefix reuse,
      bytes moved, eviction behavior, and latency pressure.
    </p>
  </header>
  <main>
    <section class="summary">{summary}</section>
    <section class="cards">{cards}</section>
    <nav class="tabs">
      <span class="tab">Overview</span><span class="tab">Topology</span>
      <span class="tab">Memory Tiers</span><span class="tab">Prefix Reuse</span>
      <span class="tab">Movement</span><span class="tab">Energy</span>
      <span class="tab">Eviction</span><span class="tab">Trace Replay</span>
    </nav>
    {images_by_tab}
    {rows}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return output_path


def _metric_cards(results: pd.DataFrame) -> str:
    if results.empty:
        return ""
    throughput_col = _first_col(results, ["estimated_throughput_score", "throughput_score"])
    dedup_col = _first_col(results, ["dedup_saved_bytes", "dedup_saved_gb"])
    hbm_col = _first_col(results, ["hbm_used_peak", "peak_hbm_gb"])
    moved_col = _first_col(results, ["bytes_moved", "bytes_moved_gb"])
    best = results.sort_values(throughput_col, ascending=False).iloc[0]
    dedup_scale = 1 if dedup_col.endswith("_gb") else 1024**3
    hbm_scale = 1 if hbm_col.endswith("_gb") else 1024**3
    moved_scale = 1 if moved_col.endswith("_gb") else 1024**3
    total_dedup = results[dedup_col].max() / dedup_scale
    peak_hbm = results[hbm_col].min() / hbm_scale
    bytes_moved = results[moved_col].min() / moved_scale
    cards = [
        ("Best Policy", str(best["policy"])),
        ("Min Peak HBM", f"{peak_hbm:.2f} GB"),
        ("Max Dedup Saved", f"{total_dedup:.2f} GB"),
        ("Min Bytes Moved", f"{bytes_moved:.2f} GB"),
    ]
    return "".join(
        f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in cards
    )


def _executive_summary(results: pd.DataFrame) -> str:
    if results.empty or "policy" not in results:
        return "<strong>Executive Summary</strong><p>No results loaded yet.</p>"
    naive = results[results["policy"].astype(str).str.contains("Naive", case=False, na=False)]
    throughput_col = _first_col(results, ["estimated_throughput_score", "throughput_score"])
    hbm_col = _first_col(results, ["hbm_used_peak", "peak_hbm_gb"])
    moved_col = _first_col(results, ["bytes_moved", "bytes_moved_gb"])
    dedup_col = _first_col(results, ["dedup_saved_bytes", "dedup_saved_gb"])
    stall_col = _first_col(results, ["simulated_stall_us", "simulated_stall_ms"])
    p99_col = _maybe_first_col(results, ["stall_p99_us", "stall_p99_ms"])
    active_hbm_col = _maybe_first_col(results, ["reserved_active_hbm_bytes", "active_hbm_gb"])
    best = results.sort_values(throughput_col, ascending=False).iloc[0]
    hbm = (
        results[hbm_col].min() / max(float(naive[hbm_col].max() or 1), 1) if not naive.empty else 0
    )
    movement = (
        results[moved_col].min() / max(float(naive[moved_col].max() or 1), 1)
        if not naive.empty
        else 0
    )
    dedup = results[dedup_col].max() / (1 if dedup_col.endswith("_gb") else 1024**3)
    stall = (
        results[stall_col].min() / max(float(naive[stall_col].max() or 1), 1)
        if not naive.empty
        else 0
    )
    p99 = (
        results[p99_col].min() / max(float(naive[p99_col].max() or 1), 1)
        if p99_col and not naive.empty
        else None
    )
    active_hbm = (
        results[active_hbm_col].max() / (1 if active_hbm_col.endswith("_gb") else 1024**3)
        if active_hbm_col
        else None
    )
    optional = ""
    if p99 is not None:
        optional += f" Minimum p99 stall ratio vs naive max: <strong>{p99:.2f}</strong>."
    if active_hbm is not None:
        optional += f" Max active HBM reservation: <strong>{active_hbm:.2f} GB</strong>."
    return (
        "<strong>Executive Summary</strong>"
        "<p>All values are synthetic simulation outputs, not real hardware measurements.</p>"
        f"<p>Best policy by throughput proxy: <strong>{best['policy']}</strong>. "
        f"Minimum HBM ratio vs naive max: <strong>{hbm:.2f}</strong>. "
        f"Minimum movement ratio vs naive max: <strong>{movement:.2f}</strong>. "
        f"Max dedup savings: <strong>{dedup:.2f} GB</strong>. "
        f"Minimum stall proxy ratio vs naive max: <strong>{stall:.2f}</strong>."
        f"{optional}</p>"
    )


def _images_by_tab(plots_dir: Path, base: Path) -> str:
    groups = {
        "Overview": ["peak", "summary", "dashboard"],
        "Topology": ["topology", "congestion"],
        "Memory Tiers": ["tier", "hbm"],
        "Prefix Reuse": ["dedup", "prefix"],
        "Movement": ["bytes_moved", "movement", "cross_rack"],
        "Energy": ["energy"],
        "Eviction": ["eviction"],
        "Trace Replay": ["stall", "prefetch", "heat", "active_hbm"],
    }
    paths = sorted(plots_dir.glob("*.png"))
    sections: list[str] = []
    for title, keywords in groups.items():
        selected = [path for path in paths if any(keyword in path.stem for keyword in keywords)]
        if not selected:
            continue
        figures = "\n".join(
            f'<figure><img src="{_rel(path, base)}" alt="{path.stem}">'
            f"<figcaption>{path.stem.replace('_', ' ').title()}</figcaption></figure>"
            for path in selected
        )
        sections.append(
            f'<section class="panel"><h2>{title}</h2>'
            f"<p>Simulation-only {title.lower()} view.</p>"
            f'<div class="grid">{figures}</div></section>'
        )
    return "\n".join(sections)


def _first_col(results: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in results.columns:
            return name
    raise KeyError(f"none of these columns exist: {names}")


def _maybe_first_col(results: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in results.columns:
            return name
    return None


def _rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()
