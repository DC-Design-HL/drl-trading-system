import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
for log in api.get_space_runtime("chen470/drl-trading-bot").hardware:
    pass
print("Fetching logs...")
for line in api.get_space_runtime("chen470/drl-trading-bot").stage:
    pass
