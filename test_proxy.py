#!/usr/bin/env python3
"""
Test Cloudflare Workers Proxy for Binance Testnet

Run this after deploying your Cloudflare Worker to verify it works.
"""
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

def test_proxy():
    """Test the Cloudflare Workers proxy setup."""

    print("=" * 70)
    print("CLOUDFLARE WORKERS PROXY TEST")
    print("=" * 70)

    # Check if proxy URL is configured
    proxy_url = os.getenv('BINANCE_TESTNET_PROXY_URL', '').strip()

    if not proxy_url:
        print("\n❌ ERROR: BINANCE_TESTNET_PROXY_URL not set!")
        print("\nPlease set the environment variable:")
        print("  export BINANCE_TESTNET_PROXY_URL='https://your-worker.workers.dev'")
        print("\nOr add it to your .env file:")
        print("  BINANCE_TESTNET_PROXY_URL=https://your-worker.workers.dev")
        return False

    print(f"\n✅ Proxy URL configured: {proxy_url}")

    # Test 1: Simple connectivity test
    print("\n" + "-" * 70)
    print("TEST 1: Proxy Connectivity")
    print("-" * 70)

    try:
        import requests
        response = requests.get(f"{proxy_url}/api/v3/time", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Proxy is working!")
            print(f"   Server time: {data.get('serverTime')}")
        else:
            print(f"❌ Proxy returned error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to proxy: {e}")
        return False

    # Test 2: BinanceConnector with proxy
    print("\n" + "-" * 70)
    print("TEST 2: BinanceConnector with Proxy")
    print("-" * 70)

    try:
        from src.api.binance import BinanceConnector

        # Get API keys
        api_key = os.getenv('BINANCE_TESTNET_API_KEY')
        api_secret = os.getenv('BINANCE_TESTNET_API_SECRET')

        if not api_key or not api_secret:
            print("⚠️ WARNING: API keys not found, testing public endpoints only")
            api_key = 'dummy_key'
            api_secret = 'dummy_secret'

        # Force use of legacy testnet (testnet.binance.vision)
        os.environ['USE_LEGACY_TESTNET'] = 'true'

        connector = BinanceConnector(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True
        )
        print("✅ BinanceConnector created successfully")

        # Test connectivity
        result = connector.test_connectivity()
        if result:
            print("✅ Connectivity test passed")
        else:
            print("⚠️ Connectivity test returned false (but this may be OK)")

        # Test ticker
        print("\n" + "-" * 70)
        print("TEST 3: Fetch BTC/USDT Ticker")
        print("-" * 70)

        ticker = connector.get_ticker('BTC/USDT')
        price = ticker.get('last', 0)
        print(f"✅ BTC Price: ${price:,.2f}")
        print(f"   Bid: ${ticker.get('bid', 0):,.2f}")
        print(f"   Ask: ${ticker.get('ask', 0):,.2f}")

        # Test account (only if real keys)
        if api_key != 'dummy_key':
            print("\n" + "-" * 70)
            print("TEST 4: Fetch Account Balances")
            print("-" * 70)

            balances = connector.get_all_balances()
            if balances:
                print(f"✅ Retrieved {len(balances)} balance entries:")
                for currency, amounts in list(balances.items())[:5]:  # Show first 5
                    total = amounts.get('total', 0)
                    print(f"   {currency}: {total:.8f}")
            else:
                print("⚠️ No balances found (account might be empty)")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\n🎉 Your Cloudflare Workers proxy is working correctly!")
    print("   You can now deploy this to HuggingFace Space.")
    print("\n📝 Next steps:")
    print("   1. Add BINANCE_TESTNET_PROXY_URL to HuggingFace Secrets")
    print("   2. Restart your HuggingFace Space")
    print("   3. Test the testnet tab in the dashboard")
    print("=" * 70)

    return True

if __name__ == "__main__":
    try:
        # Load .env file if exists
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            # Load manually
            env_file = Path(__file__).parent / '.env'
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        if line.strip() and not line.startswith('#') and '=' in line:
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value

        success = test_proxy()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
