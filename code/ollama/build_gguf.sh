#!/bin/bash
# Costruisce il GGUF Q4_K_M per Ollama a partire da base + adattatore LoRA (NON è l'oggetto valutato nell'articolo: va ri-valutato, vedi eval_gguf.py).
# Perché non "ADAPTER" in Ollama: nei nostri test con Ollama 0.40.6 non era utilizzabile; e il merge bf16 perde identità (articolo §3.5).
# Passi (richiede ~60 GB di disco, 16 GB RAM, nessuna GPU; ~40 min CPU):
#   1. scarica Qwen/Qwen3-14B alla revisione 40c069824f4251a91eefaf281ebe4c544efd3e18 in $BASE
#   2. scarica l'adattatore (HF: odelpa/qwen14-scienziato-v2, cartella adapter/) in $ADP
#   3. llama.cpp b11980 (binari + convert_hf_to_gguf.py) in $LLAMA
set -e
BASE=${BASE:-./base}; ADP=${ADP:-./adapter}; LLAMA=${LLAMA:-./llama.cpp}; OUT=${OUT:-./out}; mkdir -p "$OUT"
# 1) merge: somma in fp32, salvataggio fp16 (NON bf16)
python3 merge_fp16.py "$BASE" "$ADP" "$OUT/merged_f16" 1.0
# 2) conversione in GGUF f16 e quantizzazione Q4_K_M
python3 "$LLAMA/convert_hf_to_gguf.py" "$OUT/merged_f16" --outtype f16 --outfile "$OUT/q14v2_f16.gguf"
"$LLAMA/llama-quantize" "$OUT/q14v2_f16.gguf" "$OUT/q14v2_Q4_K_M.gguf" Q4_K_M
# 3) test: server locale + valutazione identità (50 domande) e GSM8K (150) con lo stesso scorer dell'articolo
"$LLAMA/llama-server" -m "$OUT/q14v2_Q4_K_M.gguf" -c 8192 -np 4 -t 8 --port 8099 --jinja --host 127.0.0.1 &
sleep 60
python3 eval_gguf.py http://127.0.0.1:8099 ../../data/sets "$OUT/gguf_eval.json" ident_test,gsm_test   # sets/ = R2/sets del dataset HF
# 4) Ollama (Modelfile in questa cartella; copiare il GGUF accanto al Modelfile)
#    ollama create ollamagdp/qwen14-scienziato-v2 -f Modelfile && ollama push ollamagdp/qwen14-scienziato-v2
