"""
Client-side Binance Testnet Integration
Makes API calls from user's browser to bypass HuggingFace geo-restrictions
"""

import streamlit as st
import streamlit.components.v1 as components


def render_testnet_tab(api_key: str, api_secret: str):
    """
    Render testnet tab with client-side JavaScript API calls.

    This bypasses HuggingFace's geo-restrictions by making API calls
    from the user's browser instead of the server.
    """

    # JavaScript/HTML component for client-side API calls
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/crypto-js@4.1.1/crypto-js.js"></script>
        <style>
            body {{
                font-family: 'Source Sans Pro', sans-serif;
                padding: 20px;
                background-color: #0E1117;
                color: #FAFAFA;
            }}
            .portfolio-card {{
                background-color: #262730;
                border-radius: 8px;
                padding: 20px;
                margin: 10px 0;
                border: 1px solid #464646;
            }}
            .metric {{
                text-align: center;
                padding: 15px;
            }}
            .metric-label {{
                font-size: 14px;
                color: #A0A0A0;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 28px;
                font-weight: bold;
                color: #FAFAFA;
            }}
            .metric-delta {{
                font-size: 14px;
                margin-top: 5px;
            }}
            .positive {{ color: #00D26A; }}
            .negative {{ color: #FF4B4B; }}
            .button {{
                padding: 10px 20px;
                margin: 5px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
            }}
            .buy-btn {{
                background-color: #00D26A;
                color: white;
            }}
            .sell-btn {{
                background-color: #FF4B4B;
                color: white;
            }}
            .info-btn {{
                background-color: #4A4A4A;
                color: white;
            }}
            #status {{
                padding: 10px;
                margin: 10px 0;
                border-radius: 4px;
                display: none;
            }}
            .status-success {{
                background-color: #00D26A20;
                border: 1px solid #00D26A;
                color: #00D26A;
            }}
            .status-error {{
                background-color: #FF4B4B20;
                border: 1px solid #FF4B4B;
                color: #FF4B4B;
            }}
            .loading {{
                text-align: center;
                padding: 20px;
                color: #A0A0A0;
            }}
        </style>
    </head>
    <body>
        <div id="status"></div>
        <div id="loading" class="loading">Loading testnet data...</div>

        <div id="portfolio" style="display:none;">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                <div class="portfolio-card metric">
                    <div class="metric-label">💰 Portfolio Value</div>
                    <div class="metric-value" id="portfolioValue">$0.00</div>
                    <div class="metric-delta" id="portfolioDelta"></div>
                </div>
                <div class="portfolio-card metric">
                    <div class="metric-label">💵 USDT Balance</div>
                    <div class="metric-value" id="usdtBalance">$0.00</div>
                </div>
                <div class="portfolio-card metric">
                    <div class="metric-label">📊 Total P&L %</div>
                    <div class="metric-value" id="totalPnl">+0.00%</div>
                    <div class="metric-delta" id="pnlDelta"></div>
                </div>
            </div>

            <div class="portfolio-card" style="margin-top: 20px;">
                <h3>📊 Current Positions</h3>
                <div id="positions"></div>
            </div>

            <div class="portfolio-card" style="margin-top: 20px;">
                <h3>🎮 Trading Controls</h3>
                <select id="symbol" style="padding: 10px; margin: 10px 0; width: 200px;">
                    <option value="BTCUSDT">BTC/USDT</option>
                    <option value="ETHUSDT">ETH/USDT</option>
                    <option value="SOLUSDT">SOL/USDT</option>
                    <option value="XRPUSDT">XRP/USDT</option>
                </select>
                <input type="number" id="amount" placeholder="Amount (USDT)" value="100" style="padding: 10px; margin: 10px; width: 200px;">
                <br>
                <button class="button buy-btn" onclick="placeBuyOrder()">🟢 BUY (Market)</button>
                <button class="button sell-btn" onclick="placeSellOrder()">🔴 SELL (Market)</button>
                <button class="button info-btn" onclick="refreshData()">🔄 Refresh</button>
            </div>
        </div>

        <script>
            const API_KEY = '{api_key}';
            const API_SECRET = '{api_secret}';
            const BASE_URL = 'https://testnet.binance.vision';

            // Generate HMAC SHA256 signature
            function generateSignature(queryString) {{
                return CryptoJS.HmacSHA256(queryString, API_SECRET).toString();
            }}

            // Make authenticated API request
            async function binanceRequest(endpoint, params = {{}}, method = 'GET') {{
                const timestamp = Date.now();
                params.timestamp = timestamp;
                params.recvWindow = 10000;

                const queryString = Object.keys(params)
                    .map(key => `${{key}}=${{params[key]}}`)
                    .join('&');

                const signature = generateSignature(queryString);
                const url = `${{BASE_URL}}${{endpoint}}?${{queryString}}&signature=${{signature}}`;

                const response = await fetch(url, {{
                    method: method,
                    headers: {{
                        'X-MBX-APIKEY': API_KEY
                    }}
                }});

                if (!response.ok) {{
                    const error = await response.json();
                    throw new Error(error.msg || 'API request failed');
                }}

                return response.json();
            }}

            // Show status message
            function showStatus(message, isError = false) {{
                const status = document.getElementById('status');
                status.textContent = message;
                status.className = isError ? 'status-error' : 'status-success';
                status.style.display = 'block';
                setTimeout(() => {{ status.style.display = 'none'; }}, 5000);
            }}

            // Fetch account balance
            async function fetchBalance() {{
                try {{
                    const account = await binanceRequest('/api/v3/account');
                    return account.balances.filter(b => parseFloat(b.free) > 0 || parseFloat(b.locked) > 0);
                }} catch (error) {{
                    console.error('Error fetching balance:', error);
                    throw error;
                }}
            }}

            // Fetch ticker price
            async function fetchPrice(symbol) {{
                try {{
                    const response = await fetch(`${{BASE_URL}}/api/v3/ticker/24hr?symbol=${{symbol}}`);
                    const ticker = await response.json();
                    return parseFloat(ticker.lastPrice);
                }} catch (error) {{
                    console.error('Error fetching price:', error);
                    return 0;
                }}
            }}

            // Update portfolio display
            async function updatePortfolio() {{
                try {{
                    const balances = await fetchBalance();
                    let totalValue = 0;
                    let positions = [];

                    for (const balance of balances) {{
                        const free = parseFloat(balance.free);
                        const locked = parseFloat(balance.locked);
                        const total = free + locked;

                        if (total > 0) {{
                            if (balance.asset === 'USDT') {{
                                totalValue += total;
                                document.getElementById('usdtBalance').textContent = `$${{total.toFixed(2)}}`;
                            }} else {{
                                const price = await fetchPrice(balance.asset + 'USDT');
                                const value = total * price;
                                totalValue += value;

                                positions.push({{
                                    asset: balance.asset,
                                    amount: total.toFixed(6),
                                    price: price.toFixed(2),
                                    value: value.toFixed(2)
                                }});
                            }}
                        }}
                    }}

                    // Update portfolio metrics
                    document.getElementById('portfolioValue').textContent = `$${{totalValue.toFixed(2)}}`;

                    const initialValue = 10000;
                    const pnl = totalValue - initialValue;
                    const pnlPct = (pnl / initialValue) * 100;

                    document.getElementById('totalPnl').textContent = `${{pnlPct >= 0 ? '+' : ''}}${{pnlPct.toFixed(2)}}%`;
                    document.getElementById('totalPnl').className = pnlPct >= 0 ? 'metric-value positive' : 'metric-value negative';

                    document.getElementById('portfolioDelta').textContent = `$${{pnl >= 0 ? '+' : ''}}${{pnl.toFixed(2)}}`;
                    document.getElementById('portfolioDelta').className = pnl >= 0 ? 'metric-delta positive' : 'metric-delta negative';

                    document.getElementById('pnlDelta').textContent = `$${{pnl >= 0 ? '+' : ''}}${{pnl.toFixed(2)}}`;
                    document.getElementById('pnlDelta').className = pnl >= 0 ? 'metric-delta positive' : 'metric-delta negative';

                    // Update positions table
                    const positionsDiv = document.getElementById('positions');
                    if (positions.length === 0) {{
                        positionsDiv.innerHTML = '<p style="color: #A0A0A0;">No open positions. Portfolio is 100% USDT</p>';
                    }} else {{
                        let html = '<table style="width: 100%; border-collapse: collapse;">';
                        html += '<tr><th>Asset</th><th>Amount</th><th>Price</th><th>Value (USDT)</th></tr>';
                        positions.forEach(p => {{
                            html += `<tr><td>${{p.asset}}</td><td>${{p.amount}}</td><td>$${{p.price}}</td><td>$${{p.value}}</td></tr>`;
                        }});
                        html += '</table>';
                        positionsDiv.innerHTML = html;
                    }}

                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('portfolio').style.display = 'block';

                }} catch (error) {{
                    document.getElementById('loading').textContent = '❌ Error: ' + error.message;
                    showStatus('Error loading portfolio: ' + error.message, true);
                }}
            }}

            // Place buy order
            async function placeBuyOrder() {{
                try {{
                    const symbol = document.getElementById('symbol').value;
                    const amountUSDT = parseFloat(document.getElementById('amount').value);

                    const price = await fetchPrice(symbol);
                    const quantity = (amountUSDT / price).toFixed(6);

                    await binanceRequest('/api/v3/order', {{
                        symbol: symbol,
                        side: 'BUY',
                        type: 'MARKET',
                        quantity: quantity
                    }}, 'POST');

                    showStatus(`✅ BUY order placed: ${{quantity}} ${{symbol.replace('USDT', '')}} @ $${{price.toFixed(2)}}`);
                    setTimeout(updatePortfolio, 1000);
                }} catch (error) {{
                    showStatus('❌ Order failed: ' + error.message, true);
                }}
            }}

            // Place sell order
            async function placeSellOrder() {{
                try {{
                    const symbol = document.getElementById('symbol').value;
                    const balances = await fetchBalance();
                    const asset = symbol.replace('USDT', '');
                    const balance = balances.find(b => b.asset === asset);

                    if (!balance || parseFloat(balance.free) === 0) {{
                        showStatus(`❌ No ${{asset}} to sell`, true);
                        return;
                    }}

                    const quantity = parseFloat(balance.free).toFixed(6);

                    await binanceRequest('/api/v3/order', {{
                        symbol: symbol,
                        side: 'SELL',
                        type: 'MARKET',
                        quantity: quantity
                    }}, 'POST');

                    showStatus(`✅ SELL order placed: ${{quantity}} ${{asset}}`);
                    setTimeout(updatePortfolio, 1000);
                }} catch (error) {{
                    showStatus('❌ Order failed: ' + error.message, true);
                }}
            }}

            // Refresh data
            function refreshData() {{
                document.getElementById('loading').style.display = 'block';
                document.getElementById('portfolio').style.display = 'none';
                updatePortfolio();
            }}

            // Initialize on load
            updatePortfolio();

            // Auto-refresh every 30 seconds
            setInterval(updatePortfolio, 30000);
        </script>
    </body>
    </html>
    """

    # Render the component
    components.html(html_code, height=800, scrolling=True)
