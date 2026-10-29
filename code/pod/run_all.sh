#!/bin/bash
# Pipeline completa nel pod: P1 -> P2 -> P3 -> P4. Uso: run_all.sh [from]  (from = P1|P2|P3|P4)
cd /workspace/pod; export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1; PY=/workspace/venv/bin/python; PY2=/workspace/venv2/bin/python; FROM=${1:-P1}
log() { echo "$(date +%T) $*" >> /workspace/stage.txt; }
killsrv() { pkill -f "vllm serve" ; sleep 8; pkill -9 -f "vllm serve"; sleep 4; }
fail() { log "FAIL $1"; python3 /workspace/pod/ph.py "$2" 0 "ERRORE: $1" "" ; touch /workspace/RUN_FAILED; bash /workspace/pod/mkart.sh force >> /workspace/logs/fail.log 2>&1; exit 1; }
if [ "$FROM" = P1 ]; then
  log "P1 start"; python3 ph.py P1 0 "avvio vLLM per generazione" "P1"
  nohup bash serve.sh p1 > /dev/null 2>&1 &
  bash waitsrv.sh || fail "vllm p1 non parte" P1
  $PY p1_gen.py >> /workspace/logs/p1.log 2>&1; rc=$?; [ $rc -eq 0 ] || fail "p1_gen rc=$rc" P1
  killsrv; cp /workspace/data/train.jsonl /workspace/data/train_final.jsonl; FROM=P2
fi
if [ "$FROM" = P2 ]; then
  log "P2 start"; python3 ph.py P2 0 "avvio training" "P3"
  $PY p2_train.py >> /workspace/logs/train.log 2>&1; rc=$?; [ $rc -eq 0 ] || fail "train rc=$rc" P2
  FROM=P3
fi
if [ "$FROM" = P3 ]; then
  log "P3 start"; python3 ph.py P3 0 "scalo i LoRA e avvio vLLM" "P4"
  $PY make_loras.py >> /workspace/logs/p3.log 2>&1 || fail "make_loras" P3
  nohup bash serve.sh lora 5 ft=/workspace/lora/ft s085=/workspace/lora/s085 s07=/workspace/lora/s07 s05=/workspace/lora/s05 s03=/workspace/lora/s03 > /dev/null 2>&1 &
  bash waitsrv.sh || fail "vllm lora non parte" P3
  python3 ph.py P3 15 "dev: gen su 6 varianti" "P4"
  CONC=96 $PY gen.py base,s03,s05,s07,s085,ft ident_dev,gsm_dev,ife_dev >> /workspace/logs/p3.log 2>&1 || fail "gen dev" P3
  python3 ph.py P3 70 "dev: punteggio" "P4"
  $PY2 score.py >> /workspace/logs/p3.log 2>&1 || fail "score dev" P3
  $PY select_scale.py > /workspace/select.txt 2>&1; cat /workspace/select.txt >> /workspace/logs/p3.log
  python3 ph.py P3 100 "selezione R1 fatta: $(tail -1 /workspace/select.txt)" "P4" "$(tail -1 /workspace/select.txt)"
  FROM=P4
fi
if [ "$FROM" = P4 ]; then
  log "P4 start"; python3 ph.py P4 0 "eval finale: base/ft/scale su set completi" "chiusura"
  if ! curl -s -m 3 http://127.0.0.1:8000/v1/models > /dev/null; then
    nohup bash serve.sh lora 5 ft=/workspace/lora/ft s085=/workspace/lora/s085 s07=/workspace/lora/s07 s05=/workspace/lora/s05 s03=/workspace/lora/s03 > /dev/null 2>&1 &
    bash waitsrv.sh || fail "vllm lora non parte (P4)" P4
  fi
  VARS=$($PY - <<'PY'
import json
s = json.load(open('/workspace/select.json')); ch = (s['chosen'] or {}).get('name'); order = ['s03', 's05', 's07', 's085', 'ft']
v = ['base', 'ft']
if ch:
    if ch != 'ft': v.append(ch)
    i = order.index(ch); nb = order[i + 1] if i + 1 < len(order) else order[i - 1]
    if nb not in v: v.append(nb)
else: v.append('s05')
print(','.join(v))
PY
)
  echo "$VARS" > /workspace/final_vars.txt; log "P4 vars $VARS"
  CONC=96 $PY gen.py $VARS arc_test,gsm_test,gsm_rest,ife_test,ife_rest,ident_test,gsm_dev,ife_dev,ident_dev >> /workspace/logs/p4.log 2>&1 || fail "gen finale" P4
  python3 ph.py P4 55 "merge fisico" "chiusura"
  CH=$($PY -c "import json;c=json.load(open('/workspace/select.json'))['chosen'];print(c['name'] if c else '')")
  ALLV="$VARS"
  if [ -n "$CH" ]; then
    killsrv; $PY merge.py $CH >> /workspace/logs/p4.log 2>&1 || fail "merge" P4
    nohup bash serve.sh merged /workspace/merged > /dev/null 2>&1 &
    bash waitsrv.sh || fail "vllm merged non parte" P4
    python3 ph.py P4 70 "eval del modello fisicamente fuso ($CH)" "chiusura"
    CONC=96 $PY gen.py merged arc_test,gsm_test,gsm_rest,ife_test,ife_rest,ident_test,gsm_dev,ife_dev,ident_dev >> /workspace/logs/p4.log 2>&1 || fail "gen merged" P4
    ALLV="$VARS,merged"
  fi
  killsrv; python3 ph.py P4 90 "punteggio e statistiche" "chiusura"
  $PY2 score.py >> /workspace/logs/p4.log 2>&1 || fail "score finale" P4
  $PY final_stats.py $ALLV >> /workspace/logs/p4.log 2>&1 || fail "stats" P4
  cd /workspace; touch /workspace/DONE_PIPELINE
  bash /workspace/pod/mkart.sh >> /workspace/logs/p4.log 2>&1
  touch /workspace/DONE; python3 /workspace/pod/ph.py DONE 100 "pipeline completa, artefatti pronti: attendo download dal Mac mini (nessuna cancellazione a tempo)" "download + verifica sha256 sul Mac, poi cancellazione pod" "$(tail -1 /workspace/select.txt)"
  log "DONE (artifacts.tgz pronto; il pod NON si auto-elimina: lo cancella la guardia sul Mac solo dopo download verificato)"
fi
