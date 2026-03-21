"""
DRL Trading System - Design System
Centralized design tokens, reusable component functions, and styling utilities.
All functions return raw HTML strings for use with st.markdown(unsafe_allow_html=True).
"""

from typing import Optional, List, Any, Union


# ═══════════════════════════════════════════════════════════════════
# Color Palette — Dark Trading Theme
# ═══════════════════════════════════════════════════════════════════

BG_PRIMARY = "#0E1117"
BG_CARD = "#1A1D23"
BG_CARD_ALT = "#151B23"
ACCENT = "#3B82F6"       # electric blue
SUCCESS = "#10B981"       # green / profit
DANGER = "#EF4444"        # red / loss
WARNING = "#F59E0B"       # amber
TEXT_PRIMARY = "#FAFAFA"
TEXT_MUTED = "#9CA3AF"
BORDER = "#2D3748"

# Extended palette
BG_HOVER = "#1C2333"
ACCENT_DIM = "#1E3A5F"
SUCCESS_DIM = "#1B3A26"
DANGER_DIM = "#3A1B1B"
WARNING_DIM = "#3A2E1B"


# ═══════════════════════════════════════════════════════════════════
# Global Styles — inject once via st.markdown
# ═══════════════════════════════════════════════════════════════════

GLOBAL_CSS = f"""
<style>
    /* ═══ Foundation ═══ */
    .stApp {{
        background-color: {BG_PRIMARY};
        color: {TEXT_PRIMARY};
    }}

    /* ═══ Sidebar ═══ */
    div[data-testid="stSidebarContent"] {{
        background-color: {BG_PRIMARY};
        border-right: 1px solid {BORDER};
    }}
    div[data-testid="stSidebarContent"] .stMarkdown h3 {{
        color: {TEXT_MUTED};
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}

    /* ═══ Metric Cards (native st.metric) ═══ */
    div[data-testid="stMetric"] {{
        background: {BG_CARD_ALT};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 16px 18px;
    }}
    div[data-testid="stMetric"] label {{
        color: {TEXT_MUTED} !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 700;
    }}
    div[data-testid="stMetricDelta"] svg {{ display: none; }}

    /* ═══ Design System Card ═══ */
    .ds-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 20px 22px;
        margin-bottom: 14px;
    }}
    .ds-card-label {{
        color: {TEXT_MUTED};
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }}
    .ds-card-value {{
        font-size: 26px;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    }}
    .ds-card-delta-pos {{ color: {SUCCESS}; font-size: 13px; }}
    .ds-card-delta-neg {{ color: {DANGER}; font-size: 13px; }}

    /* ═══ Tabs ═══ */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: {TEXT_MUTED};
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-size: 13px;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {TEXT_PRIMARY};
        background-color: rgba(255,255,255,0.04);
    }}
    .stTabs [aria-selected="true"] {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 600;
        border-bottom: 2px solid {ACCENT};
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {ACCENT} !important;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}

    /* ═══ Buttons ═══ */
    .stButton > button {{
        background: {BG_CARD_ALT};
        border: 1px solid {BORDER};
        color: {TEXT_PRIMARY};
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        background: {BG_HOVER};
        border-color: {ACCENT};
        color: {TEXT_PRIMARY};
    }}
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background: {ACCENT};
        border-color: {ACCENT};
        color: {TEXT_PRIMARY};
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background: #2563EB;
        border-color: #60A5FA;
    }}

    /* ═══ Inputs / Selects ═══ */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stDateInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div {{
        background-color: {BG_CARD_ALT} !important;
        border-color: {BORDER} !important;
        color: {TEXT_PRIMARY} !important;
    }}

    /* ═══ Text Areas ═══ */
    .stTextArea textarea {{
        background-color: {BG_CARD_ALT} !important;
        border-color: {BORDER} !important;
        color: {TEXT_PRIMARY} !important;
        border-radius: 6px;
    }}

    /* ═══ Code Blocks ═══ */
    .stCodeBlock, code, pre {{
        background-color: {BG_CARD_ALT} !important;
        border: 1px solid {BORDER};
        border-radius: 6px;
    }}

    /* ═══ Expanders ═══ */
    .streamlit-expanderHeader {{
        background: {BG_CARD_ALT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        color: {TEXT_PRIMARY};
    }}
    details {{
        background: {BG_CARD_ALT};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}

    /* ═══ Dividers ═══ */
    hr {{
        border-color: {BORDER} !important;
    }}

    /* ═══ Dataframes ═══ */
    .stDataFrame {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* ═══ Alerts ═══ */
    .stAlert {{
        background: {BG_CARD_ALT};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}

    /* ═══ Monospace numbers ═══ */
    .mono {{
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
    }}

    /* ═══ Scrollbar ═══ */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {BG_PRIMARY}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #4A5568; }}

    /* ═══ Hide defaults ═══ */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
"""


# ═══════════════════════════════════════════════════════════════════
# Helper Utilities
# ═══════════════════════════════════════════════════════════════════

def _esc(text: Any) -> str:
    """HTML-escape a value, handling None safely."""
    if text is None:
        return "—"
    s = str(text)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _pnl_color(value: Optional[float]) -> str:
    """Return SUCCESS or DANGER color based on sign."""
    if value is None or value == 0:
        return TEXT_MUTED
    return SUCCESS if value > 0 else DANGER


def _pnl_sign(value: Optional[float]) -> str:
    """Return +/- prefix for a numeric value."""
    if value is None:
        return ""
    return "+" if value >= 0 else ""


def _format_number(value: Optional[float], decimals: int = 2, prefix: str = "$") -> str:
    """Format a number with commas and optional prefix, handling None."""
    if value is None:
        return "—"
    try:
        v = float(value)
        formatted = f"{abs(v):,.{decimals}f}"
        sign = "+" if v > 0 else ("-" if v < 0 else "")
        return f"{sign}{prefix}{formatted}" if prefix else f"{sign}{formatted}"
    except (ValueError, TypeError):
        return "—"


# ═══════════════════════════════════════════════════════════════════
# Reusable Component Functions
# ═══════════════════════════════════════════════════════════════════

def metric_card(
    label: str,
    value: Any,
    delta: Optional[float] = None,
    icon: Optional[str] = None,
) -> str:
    """
    Render a styled metric card.

    Args:
        label: Small uppercase label text.
        value: Main display value (string or number).
        delta: Optional delta number — green if positive, red if negative.
        icon: Optional emoji/icon prefix for the label.

    Returns:
        HTML string.
    """
    label = _esc(label) if label else "—"
    icon_html = f"{_esc(icon)} " if icon else ""

    # Format value
    if value is None:
        value_html = '<span style="color:{TEXT_MUTED};">—</span>'
    elif isinstance(value, (int, float)):
        value_html = f'<span class="mono">{_format_number(value)}</span>'
    else:
        value_html = _esc(str(value))

    # Format delta
    delta_html = ""
    if delta is not None:
        try:
            d = float(delta)
            d_color = _pnl_color(d)
            d_sign = _pnl_sign(d)
            delta_html = (
                f'<div style="color:{d_color};font-size:13px;margin-top:4px;">'
                f'{d_sign}${abs(d):,.2f}</div>'
            )
        except (ValueError, TypeError):
            pass

    return f"""
    <div class="ds-card">
        <div class="ds-card-label">{icon_html}{label}</div>
        <div class="ds-card-value">{value_html}</div>
        {delta_html}
    </div>
    """


def status_badge(text: str, color: Optional[str] = None) -> str:
    """
    Render an inline status badge (pill).

    Args:
        text: Badge text.
        color: Hex color. Defaults to TEXT_MUTED.

    Returns:
        HTML string.
    """
    if not text:
        text = "—"
    color = color or TEXT_MUTED
    # Compute a dim background from the color
    bg = color + "22"  # 13% opacity hex
    return (
        f'<span style="'
        f"background:{bg};"
        f"color:{color};"
        f"padding:3px 12px;"
        f"border-radius:4px;"
        f"font-size:11px;"
        f"font-weight:600;"
        f"letter-spacing:0.5px;"
        f'">{_esc(text)}</span>'
    )


def pnl_text(value: Optional[float]) -> str:
    """
    Render a colored +/- P&L string with monospace font.

    Args:
        value: P&L amount. None renders as "—".

    Returns:
        HTML string.
    """
    if value is None:
        return f'<span style="color:{TEXT_MUTED};font-family:monospace;">—</span>'
    try:
        v = float(value)
    except (ValueError, TypeError):
        return f'<span style="color:{TEXT_MUTED};font-family:monospace;">—</span>'

    color = _pnl_color(v)
    sign = _pnl_sign(v)
    return (
        f'<span style="color:{color};font-weight:600;font-family:monospace;">'
        f'{sign}${abs(v):,.2f}</span>'
    )


def section_header(title: str, icon: Optional[str] = None) -> str:
    """
    Render a styled section header.

    Args:
        title: Section title text.
        icon: Optional emoji prefix.

    Returns:
        HTML string.
    """
    title = _esc(title) if title else ""
    icon_html = f"{_esc(icon)} " if icon else ""
    return (
        f'<div style="'
        f"font-size:18px;"
        f"font-weight:700;"
        f"color:{TEXT_PRIMARY};"
        f"margin:20px 0 12px 0;"
        f"padding-bottom:8px;"
        f"border-bottom:2px solid {BORDER};"
        f'">{icon_html}{title}</div>'
    )


def styled_table(headers: List[str], rows: List[List[Any]]) -> str:
    """
    Render an HTML table with alternating row colors and styled headers.

    Args:
        headers: Column header strings.
        rows: List of row data (each row is a list of cell values).

    Returns:
        HTML string.
    """
    if not headers:
        return ""

    # Build header row
    th_cells = "".join(
        f'<th style="'
        f"padding:12px 16px;"
        f"text-align:left;"
        f"color:{TEXT_MUTED};"
        f"font-size:11px;"
        f"text-transform:uppercase;"
        f"letter-spacing:1px;"
        f"font-weight:600;"
        f"border-bottom:1px solid {BORDER};"
        f'">{_esc(h)}</th>'
        for h in headers
    )

    # Build body rows
    body_rows = ""
    safe_rows = rows if rows else []
    for i, row in enumerate(safe_rows):
        bg = BG_CARD if i % 2 == 0 else BG_CARD_ALT
        cells = ""
        for cell in row:
            # If cell is already HTML (contains <span), pass through; otherwise escape
            cell_str = str(cell) if cell is not None else "—"
            if "<span" in cell_str or "<div" in cell_str:
                cell_html = cell_str
            else:
                cell_html = _esc(cell_str)
            cells += (
                f'<td style="'
                f"padding:12px 16px;"
                f"color:{TEXT_PRIMARY};"
                f"font-size:13px;"
                f"border-bottom:1px solid {BORDER}22;"
                f'">{cell_html}</td>'
            )
        body_rows += f'<tr style="background:{bg};">{cells}</tr>'

    if not safe_rows:
        colspan = len(headers)
        body_rows = (
            f'<tr><td colspan="{colspan}" style="'
            f"padding:30px;text-align:center;color:{TEXT_MUTED};font-size:14px;"
            f'">No data available</td></tr>'
        )

    return (
        f'<div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;overflow-x:auto;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f"<thead><tr>{th_cells}</tr></thead>"
        f"<tbody>{body_rows}</tbody>"
        f"</table></div>"
    )


def card_container(content: str) -> str:
    """
    Wrap arbitrary HTML content in a styled card div.

    Args:
        content: Inner HTML string.

    Returns:
        HTML string.
    """
    if content is None:
        content = ""
    return (
        f'<div style="'
        f"background:{BG_CARD};"
        f"border:1px solid {BORDER};"
        f"border-radius:10px;"
        f"padding:20px 22px;"
        f"margin-bottom:14px;"
        f'">{content}</div>'
    )


# ═══════════════════════════════════════════════════════════════════
# Composite / Higher-Order Components
# ═══════════════════════════════════════════════════════════════════

def loading_card(message: str = "Loading data...") -> str:
    """Render a loading placeholder card."""
    return card_container(
        f'<div style="text-align:center;padding:20px;">'
        f'<div style="color:{ACCENT};font-size:24px;margin-bottom:8px;">⏳</div>'
        f'<div style="color:{TEXT_MUTED};font-size:13px;">{_esc(message)}</div>'
        f"</div>"
    )


def error_card(message: str = "Something went wrong", detail: Optional[str] = None) -> str:
    """Render a friendly error card (no raw tracebacks)."""
    detail_html = ""
    if detail:
        detail_html = (
            f'<div style="color:{TEXT_MUTED};font-size:11px;margin-top:6px;">'
            f'{_esc(detail)}</div>'
        )
    return card_container(
        f'<div style="text-align:center;padding:16px;">'
        f'<div style="color:{DANGER};font-size:20px;margin-bottom:6px;">⚠️</div>'
        f'<div style="color:{DANGER};font-size:14px;font-weight:600;">{_esc(message)}</div>'
        f"{detail_html}"
        f"</div>"
    )


def position_badge(position: Optional[int]) -> str:
    """Return a LONG / SHORT / FLAT badge based on position value."""
    if position is None or position == 0:
        return status_badge("FLAT", TEXT_MUTED)
    elif position > 0:
        return status_badge("● LONG", SUCCESS)
    else:
        return status_badge("● SHORT", DANGER)


def metric_row(metrics: List[dict]) -> str:
    """
    Render a horizontal row of metric cards using CSS grid.

    Args:
        metrics: List of dicts with keys: label, value, delta (optional), icon (optional).

    Returns:
        HTML string.
    """
    if not metrics:
        return ""
    n = len(metrics)
    cards = "".join(
        metric_card(
            label=m.get("label", ""),
            value=m.get("value"),
            delta=m.get("delta"),
            icon=m.get("icon"),
        )
        for m in metrics
    )
    return (
        f'<div style="display:grid;grid-template-columns:repeat({n},1fr);gap:12px;">'
        f"{cards}</div>"
    )


def progress_bar(value: float, max_value: float = 100, color: Optional[str] = None) -> str:
    """
    Render a simple horizontal progress bar.

    Args:
        value: Current value.
        max_value: Maximum value (for percentage calculation).
        color: Bar color. Auto-selects based on percentage if None.

    Returns:
        HTML string.
    """
    try:
        pct = min(100, max(0, (float(value) / float(max_value)) * 100)) if max_value else 0
    except (ValueError, TypeError, ZeroDivisionError):
        pct = 0

    if color is None:
        if pct >= 70:
            color = SUCCESS
        elif pct >= 40:
            color = WARNING
        else:
            color = DANGER

    return (
        f'<div style="width:100%;background:{BORDER};height:6px;border-radius:3px;overflow:hidden;">'
        f'<div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:3px;'
        f'transition:width 0.3s ease;"></div></div>'
    )
