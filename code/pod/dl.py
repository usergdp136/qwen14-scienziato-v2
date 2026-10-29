from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-14B", revision="40c069824f4251a91eefaf281ebe4c544efd3e18", local_dir="/workspace/base",
    allow_patterns=["config.json", "generation_config.json", "model-*.safetensors", "model.safetensors.index.json", "tokenizer*", "vocab.json", "merges.txt", "LICENSE"])
print("DL_OK", p)
g = snapshot_download("openai/gsm8k", repo_type="dataset", local_dir="/workspace/gsm8k_ds")
print("GSM_OK", g)
