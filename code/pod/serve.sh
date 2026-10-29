#!/bin/bash
# uso: serve.sh p1 | serve.sh lora <n_loras> name=path ... | serve.sh merged <dir>
cd /workspace; export HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0; M=$1; shift
if [ "$M" = p1 ]; then
exec /workspace/venv/bin/vllm serve /workspace/base --served-model-name base --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --seed 1234 --max-num-seqs 256 --enable-prefix-caching --host 127.0.0.1 --port 8000 > /workspace/logs/vllm_p1.log 2>&1
elif [ "$M" = lora ]; then N=$1; shift
exec /workspace/venv/bin/vllm serve /workspace/base --served-model-name base --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --seed 1234 --max-num-seqs 96 --enable-prefix-caching --enable-lora --max-lora-rank 16 --max-loras $N --lora-modules "$@" --host 127.0.0.1 --port 8000 > /workspace/logs/vllm_lora.log 2>&1
else D=$1
exec /workspace/venv/bin/vllm serve $D --served-model-name merged --tokenizer /workspace/base --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --seed 1234 --max-num-seqs 96 --enable-prefix-caching --host 127.0.0.1 --port 8000 > /workspace/logs/vllm_merged.log 2>&1
fi
