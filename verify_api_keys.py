#!/usr/bin/env python3
"""
Verify Binance Testnet API Keys
This script checks if your API keys are valid and working.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def verify_keys():
    """Verify API keys format and connectivity"""

    api_key = os.getenv('BINANCE_TESTNET_API_KEY', '').strip()
    api_secret = os.getenv('BINANCE_TESTNET_API_SECRET', '').strip()

    print("=" * 60)
    print("🔑 Binance Testnet API Key Verification")
    print("=" * 60)

    # Check if keys exist
    if not api_key:
        print("❌ BINANCE_TESTNET_API_KEY not found in .env")
        return False

    if not api_secret:
        print("❌ BINANCE_TESTNET_API_SECRET not found in .env")
        return False

    # Check key lengths
    print(f"\n📏 API Key Length: {len(api_key)} chars")
    print(f"📏 Secret Length: {len(api_secret)} chars")

    if len(api_key) != 64:
        print(f"⚠️  WARNING: API key should be 64 characters, got {len(api_key)}")
    else:
        print("✅ API key length is correct (64 chars)")

    if len(api_secret) != 64:
        print(f"⚠️  WARNING: Secret should be 64 characters, got {len(api_secret)}")
    else:
        print("✅ Secret length is correct (64 chars)")

    # Show key preview
    print(f"\n👀 API Key Preview:")
    print(f"   First 8:  {api_key[:8]}...")
    print(f"   Last 4:   ...{api_key[-4:]}")
    print(f"   Full key: {api_key[:4]}...{api_key[-4:]}")

    # Test connectivity
    print(f"\n🔌 Testing connection to Binance Testnet...")
    try:
        from src.api.binance import BinanceConnector

        connector = BinanceConnector(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True
        )

        print("✅ BinanceConnector created successfully")

        # Test public endpoint
        print("\n🌐 Testing public endpoint (server time)...")
        try:
            time_result = connector.exchange.fetch_time()
            print(f"✅ Server time: {time_result}")
        except Exception as e:
            print(f"⚠️  Public endpoint failed: {e}")

        # Test private endpoint (account info)
        print("\n🔐 Testing private endpoint (account info)...")
        try:
            balances = connector.get_all_balances()
            print(f"✅ Account access successful!")
            print(f"📊 Found {len(balances)} non-zero balances")

            if balances:
                print("\n💰 Balances:")
                for currency, amounts in balances.items():
                    total = amounts.get('total', 0)
                    print(f"   {currency}: {total}")
            else:
                print("⚠️  No balances found (account might be empty)")
                print("💡 Tip: Get free testnet funds at testnet.binance.vision")

            return True

        except Exception as e:
            print(f"❌ Account access failed: {e}")
            print("\n🔍 Common issues:")
            print("   1. Wrong API keys (check testnet.binance.vision)")
            print("   2. Keys not from SPOT testnet")
            print("   3. API restrictions enabled")
            return False

    except Exception as e:
        print(f"❌ Failed to create connector: {e}")
        return False

if __name__ == '__main__':
    success = verify_keys()
    print("\n" + "=" * 60)
    if success:
        print("✅ All checks passed! Keys are working.")
        print("\n📋 Next step: Add these keys to HuggingFace Secrets:")
        print("   https://huggingface.co/spaces/Chen4700/drl-trading-bot-dev/settings")
    else:
        print("❌ Some checks failed. Please review the errors above.")
    print("=" * 60)
