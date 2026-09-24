"""Self-contained HTML report for one CGM day and its nearest canonical days.

The page uses inline SVG and CSS only, so it opens offline, can be attached to
an email, and loads no third-party code. Glucose values are drawn exactly as
observed; missing readings are shown as gaps, never interpolated.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .inference import ResearchEncoder, search_manifests

_WIDTH = 720
_HEIGHT = 220
_PAD_LEFT = 44
_PAD_RIGHT = 12
_PAD_TOP = 12
_PAD_BOTTOM = 28
_Y_MIN = 40.0
_Y_MAX = 300.0
_RANGE_LOW = 70.0
_RANGE_HIGH = 180.0
_NEIGHBOR_COLORS = ("#d9480f", "#2b8a3e", "#862e9c", "#1864ab", "#c2255c")


def _read_canonical_day(path: Path) -> list[float | None]:
    values: list[float | None] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            observed = row["observed_mask"].strip() == "1"
            values.append(float(row["glucose"]) if observed else None)
    return values


def _x(index: int, count: int) -> float:
    span = _WIDTH - _PAD_LEFT - _PAD_RIGHT
    return _PAD_LEFT + span * index / max(1, count - 1)


def _y(value: float) -> float:
    clipped = min(max(value, _Y_MIN), _Y_MAX)
    span = _HEIGHT - _PAD_TOP - _PAD_BOTTOM
    return _PAD_TOP + span * (1.0 - (clipped - _Y_MIN) / (_Y_MAX - _Y_MIN))


def _path(values: Sequence[float | None]) -> str:
    """Build an SVG path that breaks at every missing reading."""

    commands = []
    pen_down = False
    for index, value in enumerate(values):
        if value is None:
            pen_down = False
            continue
        verb = "L" if pen_down else "M"
        commands.append(f"{verb}{_x(index, len(values)):.1f},{_y(value):.1f}")
        pen_down = True
    return " ".join(commands)


def _chart(
    series: Sequence[tuple[Sequence[float | None], str, float, str]],
    *,
    title: str,
) -> str:
    """Render series as (values, color, stroke width, dash) on shared axes."""

    count = max(len(values) for values, *_ in series)
    band_top = _y(_RANGE_HIGH)
    band_bottom = _y(_RANGE_LOW)
    grid = []
    for level in (70, 180, 250):
        y = _y(level)
        grid.append(
            f'<line x1="{_PAD_LEFT}" x2="{_WIDTH - _PAD_RIGHT}" y1="{y:.1f}" '
            f'y2="{y:.1f}" class="grid"/>'
            f'<text x="{_PAD_LEFT - 6}" y="{y + 4:.1f}" class="tick" '
            f'text-anchor="end">{level}</text>'
        )
    hours = []
    for hour in range(0, 25, 6):
        x = _x(round(hour * (count - 1) / 24), count)
        anchor = "end" if hour == 24 else "middle"
        hours.append(
            f'<text x="{x:.1f}" y="{_HEIGHT - 8}" class="tick" '
            f'text-anchor="{anchor}">+{hour}h</text>'
        )
    lines = []
    for values, color, width, dash in series:
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<path d="{_path(values)}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round"{dash_attribute}/>'
        )
    return (
        f'<svg viewBox="0 0 {_WIDTH} {_HEIGHT}" role="img" '
        f'aria-label="{html.escape(title)}">'
        f'<rect x="{_PAD_LEFT}" y="{band_top:.1f}" '
        f'width="{_WIDTH - _PAD_LEFT - _PAD_RIGHT}" '
        f'height="{band_bottom - band_top:.1f}" class="band"/>'
        + "".join(grid)
        + "".join(hours)
        + "".join(lines)
        + "</svg>"
    )


def _summary(values: Sequence[float | None]) -> dict[str, float]:
    observed = [value for value in values if value is not None]
    if not observed:
        return {"coverage": 0.0}
    mean = sum(observed) / len(observed)
    variance = sum((value - mean) ** 2 for value in observed) / len(observed)
    in_range = sum(_RANGE_LOW <= value <= _RANGE_HIGH for value in observed)
    return {
        "coverage": len(observed) / len(values),
        "mean": mean,
        "sd": variance**0.5,
        "in_range": in_range / len(observed),
    }


def _stats_cells(values: Sequence[float | None]) -> str:
    stats = _summary(values)
    if "mean" not in stats:
        return "<td>–</td><td>–</td><td>–</td><td>0%</td>"
    return (
        f"<td>{stats['mean']:.0f}</td><td>{stats['sd']:.0f}</td>"
        f"<td>{100 * stats['in_range']:.0f}%</td>"
        f"<td>{100 * stats['coverage']:.0f}%</td>"
    )


_STYLE = """
:root{--bg:#fbfaf8;--fg:#1d1d1f;--muted:#6b6b70;--card:#ffffff;--line:#e4e2dd;
--band:#e8f3ec;--query:#1d1d1f;--warn-bg:#fff4e0;--warn-fg:#7a4b00}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececef;--muted:#a0a0a8;
--card:#1c1c20;--line:#2e2e34;--band:#1d3326;--query:#f4f4f6;--warn-bg:#3a2a0e;
--warn-fg:#f5c77a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:800px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:17px;margin:32px 0 8px}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:16px;margin:12px 0}
.warn{background:var(--warn-bg);color:var(--warn-fg);border-radius:10px;
padding:10px 14px;font-size:14px}
svg{width:100%;height:auto;display:block}
svg .grid{stroke:var(--line);stroke-width:1}svg .band{fill:var(--band)}
svg .tick{fill:var(--muted);font-size:11px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:500}td.num{font-variant-numeric:tabular-nums}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;
margin-right:6px;vertical-align:baseline}
.bar{height:6px;border-radius:3px;background:var(--line);min-width:60px}
.bar>span{display:block;height:100%;border-radius:3px}
.scroll{overflow-x:auto}footer{margin-top:40px;font-size:13px}
"""


def render_report(
    query_values: Sequence[float | None],
    query_metadata: dict[str, Any],
    matches: Sequence[dict[str, Any]],
    match_values: Sequence[Sequence[float | None]],
    model_metadata: dict[str, Any],
) -> str:
    """Return a complete HTML document for a query day and its neighbours."""

    series = [(query_values, "var(--query)", 2.4, "")]
    for index, values in enumerate(match_values):
        series.append(
            (values, _NEIGHBOR_COLORS[index % len(_NEIGHBOR_COLORS)], 1.4, "4 3")
        )
    overlay = _chart(series, title="Query day with nearest days overlaid")

    rows = [
        "<tr><td><span class='swatch' style='background:var(--query)'></span>"
        "Query</td><td>–</td>" + _stats_cells(query_values) + "</tr>"
    ]
    for index, (match, values) in enumerate(zip(matches, match_values)):
        color = _NEIGHBOR_COLORS[index % len(_NEIGHBOR_COLORS)]
        similarity = float(match["similarity"])
        width = max(0.0, min(1.0, (similarity + 1.0) / 2.0)) * 100
        rows.append(
            f"<tr><td><span class='swatch' style='background:{color}'></span>"
            f"#{match['rank']} {html.escape(str(match['dataset']))} · "
            f"{html.escape(str(match['participant_id']))}</td>"
            f"<td class='num'>{similarity:.3f}"
            f"<div class='bar'><span style='width:{width:.0f}%;"
            f"background:{color}'></span></div></td>"
            + _stats_cells(values)
            + "</tr>"
        )

    small_multiples = "".join(
        "<div class='card'><div class='muted'>"
        f"#{match['rank']} · {html.escape(str(match['dataset']))} participant "
        f"{html.escape(str(match['participant_id']))} · similarity "
        f"{float(match['similarity']):.3f}</div>"
        + _chart(
            [
                (query_values, "var(--query)", 1.2, "2 3"),
                (values, _NEIGHBOR_COLORS[index % len(_NEIGHBOR_COLORS)], 2.2, ""),
            ],
            title=f"Neighbour {match['rank']} against the query",
        )
        + "</div>"
        for index, (match, values) in enumerate(zip(matches, match_values))
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    query_name = html.escape(Path(str(query_metadata["path"])).name)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GlucoTrace day report</title><style>{_STYLE}</style></head>
<body><main>
<h1>GlucoTrace day report</h1>
<div class="muted">{query_name} · window starting
{html.escape(str(query_metadata["start_time"]))} ·
{100 * float(query_metadata["observed_fraction"]):.0f}% of readings observed</div>
<p class="warn">Research software only. This report is not a medical device and
must not be used for diagnosis, treatment, dosing, alerts, or patient care.
Similarity describes this experimental embedding, not a clinical finding.</p>

<h2>Query day and its {len(matches)} nearest days</h2>
<div class="card">{overlay}
<div class="muted" style="font-size:13px;margin-top:6px">mg/dL over the
24-hour window. Shaded band: 70–180 mg/dL. Gaps are missing readings, not
zero.</div></div>

<div class="card scroll"><table>
<thead><tr><th>Day</th><th>Similarity</th><th>Mean</th><th>SD</th>
<th>70–180</th><th>Coverage</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>

<h2>Side by side</h2>
{small_multiples}

<footer class="muted">Generated {generated} by GlucoTrace ·
checkpoint {html.escape(str(model_metadata["checkpoint_sha256"])[:12])} ·
{int(model_metadata["embedding_dimension"])}-number fingerprint ·
calibration {html.escape(str(model_metadata["calibration_method"]))}</footer>
</main></body></html>
"""


def build_report(
    encoder: ResearchEncoder,
    query_csv: str | Path,
    manifests: Sequence[str | Path],
    *,
    top_k: int = 5,
    include_self: bool = False,
    unit: str = "mg/dL",
    window_index: int | None = None,
    timestamp_col: str = "timestamp",
    glucose_col: str = "glucose",
) -> str:
    from .data import CGMWindowDataset, load_cgm_csv

    fingerprint, metadata = encoder.encode_csv(
        query_csv,
        timestamp_col=timestamp_col,
        glucose_col=glucose_col,
        window_index=window_index,
        unit=unit,
    )
    series = load_cgm_csv(
        query_csv,
        timestamp_col=timestamp_col,
        glucose_col=glucose_col,
        interval_minutes=metadata["interval_minutes"],
        unit=unit,
    )
    window = CGMWindowDataset(
        series,
        window_size=metadata["positions"],
        stride=metadata["positions"],
        min_observed=1,
    )[metadata["window_index"]]
    query_values = [
        float(value) if observed else None
        for value, observed in zip(
            window["glucose"].tolist(), window["observed_mask"].tolist()
        )
    ]
    matches = search_manifests(
        encoder,
        fingerprint,
        manifests,
        top_k=top_k,
        exclude_sha256=None if include_self else metadata["sha256"],
    )
    match_values = [
        _read_canonical_day(Path(match["manifest"]).parent / match["source_file"])
        for match in matches
    ]
    model_metadata = {
        "checkpoint_sha256": encoder.checkpoint_sha256,
        "embedding_dimension": encoder.model.config.hidden_size,
        "calibration_method": encoder.calibration["method"],
    }
    return render_report(query_values, metadata, matches, match_values, model_metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write an offline HTML report of a CGM day and its nearest days."
    )
    parser.add_argument("query_csv", type=Path)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--glucose-col", default="glucose")
    parser.add_argument("--window-index", type=int)
    parser.add_argument("--unit", choices=("mg/dL", "mmol/L"), default="mg/dL")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="defaults to $GLUCOTRACE_CHECKPOINT, ./checkpoints, then the "
        "download cache",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("glucotrace-report.html"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    encoder = ResearchEncoder.load(args.checkpoint, device=args.device)
    document = build_report(
        encoder,
        args.query_csv,
        args.manifest,
        top_k=args.top_k,
        include_self=args.include_self,
        unit=args.unit,
        window_index=args.window_index,
        timestamp_col=args.timestamp_col,
        glucose_col=args.glucose_col,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(json.dumps({"saved": str(args.output)}))


if __name__ == "__main__":
    main()
