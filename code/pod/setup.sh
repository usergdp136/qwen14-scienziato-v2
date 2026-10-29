#!/bin/bash
# Dentro il pod (nohup). Uso: setup.sh <max_secondi_deadman>   -- P0 del run qwen14_v2_r2.
# Il watchdog qui sotto e' SOLO un deadman di ultima istanza (se il Mac/la guardia sono spenti): scatta oltre il tetto rigido (min(8h, 12$/tariffa)), mai dopo DONE.
MAX_S=${1:?}; cd /workspace; mkdir -p /workspace/results /workspace/logs /workspace/data
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(POD_ID|API_KEY)=" | xargs)
( sleep "$MAX_S"; echo "$(date) WATCHDOG POD: tetto tempo, rimuovo $RUNPOD_POD_ID" >> /workspace/watchdog.log
  runpodctl remove pod "$RUNPOD_POD_ID" >> /workspace/watchdog.log 2>&1
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" -H "User-Agent: gdp-pod/1.0" >> /workspace/watchdog.log 2>&1
  kill 1 ) > /dev/null 2>&1 &
echo "watchdog max_s $MAX_S pod $RUNPOD_POD_ID" > /workspace/watchdog.log
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_TELEMETRY=1
echo "$(date +%T) start" > /workspace/stage.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > /workspace/gpu.txt
python3 -m pip install -q uv > /workspace/logs/uv.log 2>&1 || pip install -q uv >> /workspace/logs/uv.log 2>&1
( python3 -m pip install -q huggingface_hub hf_transfer >> /workspace/logs/dl.log 2>&1
  python3 - >> /workspace/logs/dl.log 2>&1 <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-14B", revision="40c069824f4251a91eefaf281ebe4c544efd3e18", local_dir="/workspace/base",
    allow_patterns=["config.json", "generation_config.json", "model-*.safetensors", "model.safetensors.index.json", "tokenizer*", "vocab.json", "merges.txt", "LICENSE"])
print("DL_OK", p)
g = snapshot_download("openai/gsm8k", repo_type="dataset", local_dir="/workspace/gsm8k_ds")
print("GSM_OK", g)
PY
  touch /workspace/DL_DONE ) &
echo "$(date +%T) install vllm" > /workspace/stage.txt
uv venv /workspace/venv --python 3.12 >> /workspace/logs/pip.log 2>&1 || python3 -m venv /workspace/venv >> /workspace/logs/pip.log 2>&1
uv pip install --python /workspace/venv/bin/python vllm hf_transfer requests peft accelerate datasets pandas pyarrow --torch-backend=auto >> /workspace/logs/pip.log 2>&1 || { echo PIP_FAIL > /workspace/FAILED; exit 1; }
uv venv /workspace/venv2 --python 3.12 >> /workspace/logs/pip2.log 2>&1
uv pip install --python /workspace/venv2/bin/python "lm_eval[ifeval]" requests >> /workspace/logs/pip2.log 2>&1 || echo "PIP2_WARN" >> /workspace/logs/pip2.log
/workspace/venv/bin/python -c "import torch,vllm,peft,transformers;print('torch',torch.__version__,torch.version.cuda,torch.cuda.is_available(),'vllm',vllm.__version__,'peft',peft.__version__,'tf',transformers.__version__)" > /workspace/versions.txt 2>&1 || { echo IMPORT_FAIL > /workspace/FAILED; exit 1; }
echo "$(date +%T) attendo download base" > /workspace/stage.txt
until [ -f /workspace/DL_DONE ]; do sleep 5; done
[ -f /workspace/base/model.safetensors.index.json ] || { echo DL_FAIL > /workspace/FAILED; exit 1; }
echo "$(date +%T) SETUP_DONE" > /workspace/stage.txt; touch /workspace/SETUP_DONE
