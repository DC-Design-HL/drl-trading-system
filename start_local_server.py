#!/usr/bin/env python3
"""
Local Server Startup Script

Starts the Flask API server locally on port 5001.
Optionally creates a public ngrok tunnel so the HuggingFace UI can reach it.

Usage:
    python start_local_server.py              # Local only (http://127.0.0.1:5001)
    python start_local_server.py --tunnel     # With ngrok tunnel (prints public URL)
    python start_local_server.py --port 5001  # Custom port

After starting with --tunnel, copy the printed public URL into your
HuggingFace Space secrets as:  API_SERVER_URL = https://xxx.ngrok-free.app
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

# Load .env so api_server can reach storage/exchange APIs
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def start_flask(port: int):
    """Start the Flask API server in this thread."""
    from src.ui.api_server import app
    print(f"[server] Flask API starting on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


def start_tunnel(port: int) -> str:
    """Open an ngrok tunnel and return the public URL."""
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        print("[tunnel] ERROR: pyngrok not installed. Run: pip install pyngrok")
        sys.exit(1)

    # Use NGROK_AUTHTOKEN from .env if set
    auth_token = os.environ.get('NGROK_AUTHTOKEN', '')
    if auth_token:
        conf.get_default().auth_token = auth_token
    else:
        print("[tunnel] WARNING: NGROK_AUTHTOKEN not set. Free ngrok requires authentication.")
        print("[tunnel]          Set NGROK_AUTHTOKEN in .env or visit https://dashboard.ngrok.com/")

    tunnel = ngrok.connect(port, "http")
    public_url = tunnel.public_url
    # ngrok always gives http; upgrade to https
    if public_url.startswith("http://"):
        public_url = "https://" + public_url[7:]
    return public_url


def main():
    parser = argparse.ArgumentParser(description='Start local DRL trading API server')
    parser.add_argument('--tunnel', action='store_true', help='Create ngrok tunnel for HF access')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind (default: 5001)')
    args = parser.parse_args()

    port = args.port

    if args.tunnel:
        print("[tunnel] Opening ngrok tunnel...")
        public_url = start_tunnel(port)
        print()
        print("=" * 60)
        print(f"  PUBLIC URL: {public_url}")
        print()
        print("  Copy this into HuggingFace Space secrets:")
        print(f"    API_SERVER_URL = {public_url}")
        print("=" * 60)
        print()
        # Write URL to .tunnel_url for convenience
        Path('.tunnel_url').write_text(public_url)
        print(f"[tunnel] URL also saved to .tunnel_url")
    else:
        public_url = None
        print(f"[server] Starting local-only server (no tunnel)")
        print(f"[server] To expose publicly, use: python start_local_server.py --tunnel")

    # Start Flask in the main thread (blocks)
    print(f"[server] API will be at: http://127.0.0.1:{port}")
    if public_url:
        print(f"[server] Public URL:      {public_url}")
    print()
    start_flask(port)


if __name__ == '__main__':
    main()
