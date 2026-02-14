"""
Chart Components
TradingView-style candlestick charts with signal overlays.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import json


def create_candlestick_chart(
    df: pd.DataFrame,
    signals: Optional[List[Dict]] = None,
    title: str = "BTC/USDT",
    height: int = 600,
    show_volume: bool = True,
) -> go.Figure:
    """
    Create a TradingView-style candlestick chart with volume.
    
    Args:
        df: DataFrame with OHLCV data (index=timestamp)
        signals: List of trade signals [{'type': 'buy'/'sell', 'price': x, 'time': t}]
        title: Chart title
        height: Chart height in pixels
        show_volume: Whether to show volume subplot
        
    Returns:
        Plotly figure
    """
    # Create subplot with shared x-axis
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )
    else:
        fig = make_subplots(rows=1, cols=1)
        
    # Ensure index is proper datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        
    # Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1,
    )
    
    # Volume bars
    if show_volume:
        colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' 
                  for _, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5,
            ),
            row=2, col=1,
        )
        
    # Add buy/sell signals
    if signals:
        buy_signals = [s for s in signals if s.get('type') == 'buy' or s.get('position') == 1]
        sell_signals = [s for s in signals if s.get('type') == 'sell' or s.get('position') == -1]
        
        # Buy markers
        if buy_signals:
            fig.add_trace(
                go.Scatter(
                    x=[s.get('time', s.get('timestamp')) for s in buy_signals],
                    y=[s.get('price', s.get('entry_price')) for s in buy_signals],
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='#26a69a',
                        line=dict(color='white', width=1),
                    ),
                ),
                row=1, col=1,
            )
            
        # Sell markers
        if sell_signals:
            fig.add_trace(
                go.Scatter(
                    x=[s.get('time', s.get('timestamp')) for s in sell_signals],
                    y=[s.get('price', s.get('entry_price')) for s in sell_signals],
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='#ef5350',
                        line=dict(color='white', width=1),
                    ),
                ),
                row=1, col=1,
            )
            
    # Update layout for dark theme (TradingView style)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='white'),
        ),
        template='plotly_dark',
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)',
        ),
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
    )
    
    # Grid styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#1e222d',
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#1e222d',
    )
    
    return fig


def create_equity_chart(
    portfolio_values: List[float],
    timestamps: Optional[List] = None,
    initial_balance: float = 10000,
    height: int = 300,
) -> go.Figure:
    """
    Create an equity curve chart.
    
    Args:
        portfolio_values: List of portfolio values over time
        timestamps: Optional timestamps for x-axis
        initial_balance: Starting balance for comparison
        height: Chart height
        
    Returns:
        Plotly figure
    """
    if timestamps is None:
        timestamps = list(range(len(portfolio_values)))
        
    fig = go.Figure()
    
    # Portfolio value line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#26a69a', width=2),
            fill='tozeroy',
            fillcolor='rgba(38, 166, 154, 0.1)',
        )
    )
    
    # Initial balance reference line
    fig.add_hline(
        y=initial_balance,
        line_dash="dash",
        line_color="#888",
        annotation_text="Initial Balance",
    )
    
    fig.update_layout(
        title="Portfolio Value",
        template='plotly_dark',
        height=height,
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        xaxis_title="Time",
        yaxis_title="Value ($)",
    )
    
    return fig


def create_metrics_chart(
    metrics: Dict[str, float],
    height: int = 200,
) -> go.Figure:
    """
    Create a gauge chart for key metrics.
    
    Args:
        metrics: Dictionary with metric values
        height: Chart height
        
    Returns:
        Plotly figure with gauge
    """
    sharpe = metrics.get('sharpe_ratio', 0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sharpe,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sharpe Ratio", 'font': {'size': 16, 'color': 'white'}},
        delta={'reference': 0.5, 'increasing': {'color': "#26a69a"}},
        gauge={
            'axis': {'range': [-2, 4], 'tickcolor': "white"},
            'bar': {'color': "#26a69a" if sharpe > 0 else "#ef5350"},
            'bgcolor': "#1e222d",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [-2, 0], 'color': 'rgba(239, 83, 80, 0.3)'},
                {'range': [0, 0.5], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [0.5, 4], 'color': 'rgba(38, 166, 154, 0.3)'},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))
    
    fig.update_layout(
        height=height,
        paper_bgcolor='#131722',
        font={'color': 'white'},
    )
    
    return fig


def create_lightweight_chart_html(
    df: pd.DataFrame,
    signals: Optional[List[Dict]] = None,
) -> str:
    """
    Generate HTML for Lightweight Charts (TradingView library).
    
    Args:
        df: OHLCV DataFrame
        signals: Trade signals
        
    Returns:
        HTML string with embedded chart
    """
    # Convert DataFrame to format expected by lightweight-charts
    candle_data = []
    for timestamp, row in df.iterrows():
        candle_data.append({
            'time': int(timestamp.timestamp()),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
        })
        
    volume_data = []
    for timestamp, row in df.iterrows():
        color = 'rgba(38, 166, 154, 0.5)' if row['close'] >= row['open'] else 'rgba(239, 83, 80, 0.5)'
        volume_data.append({
            'time': int(timestamp.timestamp()),
            'value': float(row['volume']),
            'color': color,
        })
        
    # Prepare markers
    markers = []
    if signals:
        for s in signals:
            marker = {
                'time': int(pd.Timestamp(s.get('time', s.get('timestamp'))).timestamp()),
                'position': 'belowBar' if s.get('type') == 'buy' or s.get('position') == 1 else 'aboveBar',
                'color': '#26a69a' if s.get('type') == 'buy' or s.get('position') == 1 else '#ef5350',
                'shape': 'arrowUp' if s.get('type') == 'buy' or s.get('position') == 1 else 'arrowDown',
                'text': 'BUY' if s.get('type') == 'buy' or s.get('position') == 1 else 'SELL',
            }
            markers.append(marker)
            
    html = f"""
    <div id="chart" style="width: 100%; height: 500px;"></div>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <script>
        const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
            layout: {{
                background: {{ type: 'solid', color: '#131722' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#1e222d' }},
                horzLines: {{ color: '#1e222d' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
            }},
            rightPriceScale: {{
                borderColor: '#2B2B43',
            }},
            timeScale: {{
                borderColor: '#2B2B43',
                timeVisible: true,
            }},
        }});
        
        const candlestickSeries = chart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderDownColor: '#ef5350',
            borderUpColor: '#26a69a',
            wickDownColor: '#ef5350',
            wickUpColor: '#26a69a',
        }});
        
        candlestickSeries.setData({json.dumps(candle_data)});
        
        const markers = {json.dumps(markers)};
        if (markers.length > 0) {{
            candlestickSeries.setMarkers(markers);
        }}
        
        const volumeSeries = chart.addHistogramSeries({{
            priceFormat: {{
                type: 'volume',
            }},
            priceScaleId: '',
            scaleMargins: {{
                top: 0.8,
                bottom: 0,
            }},
        }});
        
        volumeSeries.setData({json.dumps(volume_data)});
        
        chart.timeScale().fitContent();
    </script>
    """
    
    return html
