import re

with open("src/features/whale_tracker.py", "r") as f:
    content = f.read()

# 1. BinanceLiquidationTracker
content = re.sub(
    r'    def _get_request_config\(self, endpoint: str.*?    def _get_okx_data',
    '    def _get_okx_data',
    content,
    flags=re.DOTALL
)

liquidation_replacement = '''    def get_liquidation_stats(self) -> Dict:
        """Get liquidation statistics using funding rate and open interest as proxy (from OKX fallback)."""
        if 'stats' in self.cache:
            cached_time, cached_data = self.cache['stats']
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
                
        try:
            okx_data = self._get_okx_data()
            funding_sentiment = -np.sign(okx_data['funding_rate']) * min(abs(okx_data['funding_rate']) * 100, 1)
            imbalance = okx_data['imbalance']
            
            stats = {
                'long_ratio': okx_data['long_ratio'],
                'short_ratio': okx_data['short_ratio'],
                'funding_rate': okx_data['funding_rate'],
                'open_interest': 0,
                'imbalance': np.clip(imbalance + funding_sentiment * 0.3, -1, 1),
                'long_liquidations': 0,
                'short_liquidations': 0,
                'total_liquidations': 0,
                'count': 0
            }
            self.cache['stats'] = (time.time(), stats)
            return stats
        except Exception as e:
            return {
                'long_ratio': 0.5, 'short_ratio': 0.5, 'funding_rate': 0,
                'open_interest': 0, 'imbalance': 0, 'long_liquidations': 0,
                'short_liquidations': 0, 'total_liquidations': 0, 'count': 0
            }'''

content = re.sub(
    r'    def get_liquidation_stats\(self\) -> Dict:.*?        except Exception as e:.*?\n            }.*?\n',
    liquidation_replacement + '\n',
    content,
    flags=re.DOTALL
)

# 2. BinanceOITracker
content = re.sub(
    r'    def _get_request_config\(self, endpoint: str.*?    def get_oi_signal',
    '    def get_oi_signal',
    content,
    flags=re.DOTALL
)

oi_replacement = '''    def get_oi_signal(self) -> Dict:
        """Get Open Interest trend signal using OKX."""
        if 'signal' in self.cache:
            cached_time, cached_data = self.cache['signal']
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
                
        try:
            okx_symbol = self.symbol.replace("USDT", "-USDT-SWAP")
            url = "https://www.okx.com/api/v5/public/open-interest"
            response = requests.get(url, params={"instId": okx_symbol, "instType": "SWAP"}, timeout=5)
            
            current_oi = 0.0
            if response.status_code == 200:
                data = response.json().get('data', [])
                if data:
                    current_oi = float(data[0].get('oi', 0))
                    
            price_url = "https://www.okx.com/api/v5/market/ticker"
            response = requests.get(price_url, params={"instId": okx_symbol}, timeout=5)
            
            current_price = 0.0
            if response.status_code == 200:
                data = response.json().get('data', [])
                if data:
                    current_price = float(data[0].get('last', 0))
                    
            self.oi_history.append({'oi': current_oi, 'price': current_price, 'time': time.time()})
            
            signal = 0.0
            if len(self.oi_history) >= 2:
                prev = self.oi_history[-2]
                oi_change = (current_oi - prev['oi']) / prev['oi'] if prev['oi'] > 0 else 0
                price_change = (current_price - prev['price']) / prev['price'] if prev['price'] > 0 else 0
                if oi_change > 0.001:
                    signal = 1.0 if price_change > 0 else -1.0
                elif oi_change < -0.001:
                    signal = 0.0
                    
            result = {'oi': current_oi, 'price': current_price, 'signal': np.clip(signal, -1, 1)}
            self.cache['signal'] = (time.time(), result)
            return result
            
        except Exception as e:
            return {'oi': 0, 'price': 0, 'signal': 0}'''

content = re.sub(
    r'    def get_oi_signal\(self\) -> Dict:.*?        except Exception as e:.*?\n            return {\'oi\': 0, \'price\': 0, \'signal\': 0}\n',
    oi_replacement + '\n',
    content,
    flags=re.DOTALL
)

# 3. BinanceTopTraderClient
content = re.sub(
    r'    def _get_request_config\(self, endpoint: str.*?    def get_large_transactions',
    '    def get_large_transactions',
    content,
    flags=re.DOTALL
)

top_replacement = '''    def get_large_transactions(self, min_btc: float = 100) -> Dict:
        """Get top trader positioning signal using OKX Top Trader ratio."""
        if 'top_trader' in self.cache:
            cached_time, cached_data = self.cache['top_trader']
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
        
        try:
            ccy = self.symbol.replace("USDT", "")
            url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio"
            response = requests.get(url, params={"ccy": ccy, "period": "1H"}, timeout=5)
            
            long_ratio = 0.5
            if response.status_code == 200:
                data = response.json().get('data', [])
                if data:
                    item = data[0]
                    if isinstance(item, list):
                        long_ratio = float(item[1])
                    elif isinstance(item, dict):
                        long_ratio = float(item.get('longAccount', item.get('ratio', 0.5)))
                        
            short_ratio = 1.0 - long_ratio
            if long_ratio > 0.55:
                bias = 'bullish'
                strength = min((long_ratio - 0.5) * 2, 1.0)
            elif short_ratio > 0.55:
                bias = 'bearish'
                strength = min((short_ratio - 0.5) * 2, 1.0)
            else:
                bias = 'neutral'
                strength = 0.0
                
            result = {
                'long_ratio': long_ratio,
                'short_ratio': short_ratio,
                'bias': bias,
                'strength': strength
            }
            self.cache['top_trader'] = (time.time(), result)
            return result
        except Exception as e:
            return {'long_ratio': 0.5, 'short_ratio': 0.5, 'bias': 'neutral', 'strength': 0.0}'''

content = re.sub(
    r'    def get_large_transactions\(self,.*?        except Exception as e:.*?\n.*?\n',
    top_replacement + '\n\n',
    content,
    flags=re.DOTALL
)

with open("src/features/whale_tracker.py", "w") as f:
    f.write(content)
