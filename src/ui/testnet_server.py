"""
Server-side Binance Testnet Integration
Uses BinanceConnector on the server (not client browser)
"""

import streamlit as st
import os
from src.api.binance import BinanceConnector
import pandas as pd


def render_testnet_tab_server(api_key: str, api_secret: str):
    """
    Render testnet tab with server-side API calls.
    """

    # Force use of legacy testnet
    os.environ['USE_LEGACY_TESTNET'] = 'true'

    # Show platform info
    import platform
    is_huggingface = os.path.exists('/app')  # HuggingFace Spaces use /app directory

    if is_huggingface:
        st.info("""
        💡 **Testnet Tab - Local Development Feature**

        This tab is designed for **local development** but has limitations on HuggingFace:
        - Binance geo-blocks HuggingFace's datacenter (HTTP 451)
        - External proxies can't be reached (DNS restrictions)

        ✅ **Works locally** when you run: `streamlit run src/ui/app.py`
        """)

    try:
        # Strip whitespace from API credentials (defensive)
        api_key = api_key.strip() if api_key else ''
        api_secret = api_secret.strip() if api_secret else ''

        # Validate API keys
        if not api_key or not api_secret:
            st.error("❌ API keys are empty after sanitization!")
            return

        if len(api_key) < 20 or len(api_secret) < 20:
            st.error(f"❌ API keys seem too short (key: {len(api_key)}, secret: {len(api_secret)})")
            return

        # Debug: Show configuration
        proxy_url = os.getenv('BINANCE_TESTNET_PROXY_URL', '').strip()
        if proxy_url:
            st.success(f"🌐 Using Cloudflare Proxy: {proxy_url}")
        else:
            st.warning("⚠️ No proxy configured - direct connection (may be geo-blocked)")

        st.info(f"🔑 API Key Length: {len(api_key)} chars (first 8: {api_key[:8]}..., last 4: ...{api_key[-4:]})")
        st.info(f"🔐 Secret Length: {len(api_secret)} chars")

        # Try to create connector (prefer ccxt)
        testnet = None
        use_http_proxy = False

        try:
            testnet = BinanceConnector(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True
            )
            st.success("✅ BinanceConnector (ccxt) created successfully")
        except Exception as e:
            st.warning(f"⚠️ BinanceConnector failed: {e}")
            st.info("🔄 Trying HTTP proxy fallback...")

            # Try HTTP proxy fallback
            try:
                from src.api.http_proxy import BinanceHTTPProxy
                testnet = BinanceHTTPProxy(
                    api_key=api_key,
                    api_secret=api_secret
                )
                use_http_proxy = True
                st.success("✅ Using HTTP Proxy fallback")
            except Exception as e2:
                st.error(f"❌ HTTP Proxy also failed: {e2}")
                import traceback
                st.code(traceback.format_exc())
                return

        # Test connectivity
        try:
            connectivity = testnet.test_connectivity()
            if connectivity:
                connector_type = "HTTP Proxy" if use_http_proxy else "ccxt"
                st.success(f"✅ Connected to Binance Testnet ({connector_type})")
            else:
                st.warning("⚠️ Connectivity test returned false, but trying anyway...")
        except Exception as e:
            st.error(f"❌ Connectivity test failed with error: {e}")
            import traceback
            st.code(traceback.format_exc())

        # Get balances
        try:
            balances = testnet.get_all_balances()
            st.info(f"📊 Retrieved {len(balances) if balances else 0} balance entries")
        except Exception as e:
            error_str = str(e)

            # Check if it's a geo-restriction or DNS error
            if "451" in error_str or "restricted location" in error_str:
                st.error("🚫 Binance Testnet is geo-blocked from HuggingFace servers")
                st.warning("""
                **HuggingFace Limitation:**

                Binance blocks access from HuggingFace's datacenter location (HTTP 451 - Legal Restriction).

                **Options:**
                1. **Run locally** - Clone the repo and run on your machine
                2. **Use Live Trading** - The main dashboard works with live data
                3. **Deploy elsewhere** - Use Render, Railway, or other platforms

                This is a Binance geo-restriction, not a bug in the code.
                """)
                return

            elif "gaierror" in error_str or "Failed to resolve" in error_str or "No address associated" in error_str:
                st.error("🌐 HuggingFace cannot reach external proxy (DNS restriction)")
                st.warning("""
                **HuggingFace Limitation:**

                HuggingFace Spaces cannot resolve DNS for Cloudflare Workers or external proxies.

                **Options:**
                1. **Run locally** - Works perfectly on your local machine
                2. **Use Live Trading** - Main features work without testnet
                3. **Different platform** - Deploy on Render/Railway/etc.

                This is a HuggingFace network policy, not a code issue.
                """)
                return

            else:
                # Unknown error
                st.error(f"❌ Failed to get balances: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

        if not balances:
            st.error("❌ No balances found. Account might be empty or API keys invalid.")
            st.warning("💡 Tip: Go to testnet.binance.vision and claim test funds")
            return

        # Calculate portfolio value
        portfolio_value = 0.0
        positions_data = []

        for currency, amounts in balances.items():
            total = float(amounts.get('total', 0))
            if total > 0:
                if currency == 'USDT':
                    portfolio_value += total
                else:
                    # Get current price
                    try:
                        ticker = testnet.get_ticker(f"{currency}/USDT")
                        price = float(ticker.get('last', 0))
                        if price > 0:
                            value_usdt = total * price
                            portfolio_value += value_usdt

                            positions_data.append({
                                'Asset': currency,
                                'Amount': f"{total:.6f}",
                                'Price': f"${price:,.2f}",
                                'Value (USDT)': f"${value_usdt:,.2f}"
                            })
                    except Exception as e:
                        # Skip assets that don't have USDT pairs
                        pass

        # Portfolio overview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="💰 Portfolio Value",
                value=f"${portfolio_value:,.2f}",
                delta=f"{portfolio_value - 10000:.2f} USDT" if portfolio_value != 10000 else None
            )

        with col2:
            usdt_balance = balances.get('USDT', {}).get('free', 0)
            st.metric(
                label="💵 USDT Balance",
                value=f"${float(usdt_balance):,.2f}"
            )

        with col3:
            pnl_pct = ((portfolio_value - 10000) / 10000) * 100 if portfolio_value > 0 else 0
            st.metric(
                label="📊 Total P&L %",
                value=f"{pnl_pct:+.2f}%",
                delta=f"${portfolio_value - 10000:+,.2f}"
            )

        st.markdown("---")

        # Current positions
        if positions_data:
            st.markdown("### 📊 Current Positions")
            positions_df = pd.DataFrame(positions_data)
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions. Portfolio is 100% USDT")

        st.markdown("---")

        # Trading controls
        st.markdown("### 🎮 Trading Controls")

        col1, col2 = st.columns(2)

        with col1:
            trade_symbol = st.selectbox(
                "Select Pair",
                options=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT'],
                key="testnet_symbol"
            )

        with col2:
            trade_amount = st.number_input(
                "Amount (USDT)",
                min_value=10.0,
                max_value=float(usdt_balance) if usdt_balance else 10000.0,
                value=100.0,
                step=10.0,
                key="testnet_amount"
            )

        col1, col2, col3 = st.columns(3)

        if col1.button("🟢 BUY (Market)", key="testnet_buy", use_container_width=True):
            try:
                ticker = testnet.get_ticker(trade_symbol)
                current_price = float(ticker['last'])
                amount_base = trade_amount / current_price

                order = testnet.place_market_order(
                    symbol=trade_symbol,
                    side='buy',
                    amount=amount_base
                )

                if order:
                    st.success(f"✅ BUY order placed: {amount_base:.6f} {trade_symbol.split('/')[0]} @ ${current_price:,.2f}")
                    st.rerun()
                else:
                    st.error("❌ Order failed")
            except Exception as e:
                st.error(f"❌ Order failed: {e}")

        if col2.button("🔴 SELL (Market)", key="testnet_sell", use_container_width=True):
            try:
                base_currency = trade_symbol.split('/')[0]
                base_balance = balances.get(base_currency, {}).get('free', 0)

                if float(base_balance) == 0:
                    st.warning(f"No {base_currency} to sell")
                else:
                    order = testnet.place_market_order(
                        symbol=trade_symbol,
                        side='sell',
                        amount=float(base_balance)
                    )

                    if order:
                        st.success(f"✅ SELL order placed: {base_balance} {base_currency}")
                        st.rerun()
                    else:
                        st.error("❌ Order failed")
            except Exception as e:
                st.error(f"❌ Order failed: {e}")

        if col3.button("📋 Open Orders", key="testnet_orders", use_container_width=True):
            try:
                open_orders = testnet.get_open_orders()
                if open_orders:
                    st.write(open_orders)
                else:
                    st.info("No open orders")
            except Exception as e:
                st.error(f"❌ Failed to fetch orders: {e}")

        st.markdown("---")

        # Info boxes
        st.markdown("### ℹ️ Testnet Information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.info("""
            **About Testnet:**
            - Zero real money risk
            - Test trading strategies
            - Same API as production
            - Perfect for learning
            """)

        with info_col2:
            st.warning("""
            **Important Notes:**
            - Using testnet.binance.vision
            - Testnet funds reset periodically
            - Use for testing only
            - Get free testnet funds from Binance
            """)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())
