import os
import requests
from dotenv import load_dotenv

load_dotenv(".env")
token = os.environ.get("HF_TOKEN")
headers = {"Authorization": f"Bearer {token}"}
url = "https://huggingface.co/api/spaces/chen470/drl-trading-bot"

r = requests.get(url, headers=headers)
data = r.json()
print("Stage:", data.get('runtime', {}).get('stage'))
if 'error' in data.get('runtime', {}):
    print("Error:", data['runtime']['error'])
