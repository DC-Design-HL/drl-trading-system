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

    try:
        # Debug: Show configuration
        proxy_url = os.getenv('BINANCE_TESTNET_PROXY_URL', '').strip()
        if proxy_url:
            st.success(f"🌐 Using Cloudflare Proxy: {proxy_url}")
        else:
            st.warning("⚠️ No proxy configured - direct connection (may be geo-blocked)")

        st.info(f"🔑 Using API Key: {api_key[:4]}...{api_key[-4:]}")

        # Create connector
        try:
            testnet = BinanceConnector(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True
            )
            st.success("✅ BinanceConnector created successfully")
        except Exception as e:
            st.error(f"❌ Failed to create BinanceConnector: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

        # Test connectivity
        try:
            connectivity = testnet.test_connectivity()
            if connectivity:
                st.success("✅ Connected to Binance Testnet (testnet.binance.vision)")
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
