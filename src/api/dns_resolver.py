"""
Custom DNS Resolution for Restricted Environments
Bypasses local DNS restrictions by using alternative DNS servers or hardcoded IPs
"""

import socket
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.connection import create_connection
import logging

logger = logging.getLogger(__name__)


class CustomDNSAdapter(HTTPAdapter):
    """
    HTTP Adapter that uses custom DNS resolution.
    Useful in environments where DNS is restricted (e.g., HuggingFace Spaces)
    """

    def __init__(self, dns_mapping=None, *args, **kwargs):
        """
        Initialize adapter with custom DNS mapping.

        Args:
            dns_mapping: Dict of hostname -> IP address
                         e.g., {'example.com': '1.2.3.4'}
        """
        self.dns_mapping = dns_mapping or {}
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        """Override pool manager to use custom DNS"""
        # Store original getaddrinfo
        self._original_getaddrinfo = socket.getaddrinfo

        # Create custom getaddrinfo that uses our DNS mapping
        def custom_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            # Check if we have a custom DNS entry for this host
            if host in self.dns_mapping:
                custom_ip = self.dns_mapping[host]
                logger.info(f"🔍 Using custom DNS: {host} -> {custom_ip}")
                # Return the custom IP
                return self._original_getaddrinfo(custom_ip, port, family, type, proto, flags)
            else:
                # Use normal DNS resolution
                return self._original_getaddrinfo(host, port, family, type, proto, flags)

        # Monkey-patch socket.getaddrinfo
        socket.getaddrinfo = custom_getaddrinfo

        return super().init_poolmanager(*args, **kwargs)


def create_session_with_custom_dns(dns_mapping=None):
    """
    Create a requests session with custom DNS resolution.

    Args:
        dns_mapping: Dict of hostname -> IP address

    Returns:
        requests.Session with custom DNS adapter
    """
    session = requests.Session()
    adapter = CustomDNSAdapter(dns_mapping=dns_mapping)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


# Cloudflare Worker IP addresses (from DNS lookup)
# These IPs are for frosty-lake-46b0.chen470.workers.dev
CLOUDFLARE_WORKER_IPS = {
    'ipv4': ['172.67.155.75', '104.21.58.29'],
    'ipv6': ['2606:4700:3032::6815:3a1d', '2606:4700:3035::ac43:9b4b']
}


def get_cloudflare_worker_dns_mapping(hostname: str):
    """
    Get DNS mapping for Cloudflare Worker hostname.

    Args:
        hostname: Cloudflare Worker hostname (e.g., 'frosty-lake-46b0.chen470.workers.dev')

    Returns:
        Dict mapping hostname to IP address
    """
    # Try IPv4 first
    ip = CLOUDFLARE_WORKER_IPS['ipv4'][0]
    logger.info(f"📍 DNS mapping: {hostname} -> {ip}")
    return {hostname: ip}


def test_custom_dns():
    """Test custom DNS resolution"""
    hostname = 'frosty-lake-46b0.chen470.workers.dev'
    dns_mapping = get_cloudflare_worker_dns_mapping(hostname)

    session = create_session_with_custom_dns(dns_mapping)

    try:
        # Test with custom DNS
        url = f'https://{hostname}/api/v3/time'
        logger.info(f"Testing custom DNS: {url}")
        response = session.get(url, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ Custom DNS works! Response: {response.json()}")
        return True
    except Exception as e:
        logger.error(f"❌ Custom DNS failed: {e}")
        return False


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_custom_dns()
