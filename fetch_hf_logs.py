import os
from dotenv import load_dotenv
import requests

load_dotenv(".env")
token = os.environ.get("HF_TOKEN")

url = "https://huggingface.co/api/spaces/chen470/drl-trading-bot/logs/container"
headers = {"Authorization": f"Bearer {token}"}

response = requests.get(url, headers=headers, stream=True)
lines = []

for line in response.iter_lines():
    if line:
        decoded_line = line.decode('utf-8')
        lines.append(decoded_line)

for l in lines[-100:]:
    print(l)
