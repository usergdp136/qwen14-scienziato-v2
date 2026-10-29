#!/bin/bash
# Costruisce /workspace/artifacts.tgz (+ .sha256, FILES.sha256). Idempotente. Uso: mkart.sh [force]  (force = riscrive anche se gia' pronto: emergenza/parziale)
cd /workspace
if [ "$1" != force ] && [ -f ARTIFACTS_READY ]; then echo "gia' pronto"; exit 0; fi
rm -f ARTIFACTS_READY artifacts.tgz artifacts.tgz.tmp artifacts.tgz.sha256
LIST=""
for p in adapter lora/ft data/train.jsonl data/val.jsonl data/MANIFEST.json data/pilot.json data/prompts_local.jsonl data/*_stats.json results select.json select.txt final_stats.json final_table.md final_vars.txt metrics.jsonl train_summary.json smoke.json template_check.json logs phase.json stage.txt versions.txt gpu.txt watchdog.log DONE RUN_FAILED FAILED ABORT pod sets; do [ -e "$p" ] && LIST="$LIST $p"; done
find $LIST -type f ! -name '*.tmp' -print0 | xargs -0 sha256sum > /workspace/FILES.sha256 2>/dev/null
tar czf artifacts.tgz.tmp --exclude='*.tmp' $LIST FILES.sha256 2>> /workspace/logs/mkart.err; rc=$?
[ $rc -eq 0 ] || echo "tar rc=$rc" >> /workspace/logs/mkart.err
mv artifacts.tgz.tmp artifacts.tgz && sha256sum artifacts.tgz > artifacts.tgz.sha256 && ls -la artifacts.tgz >> /workspace/stage.txt && touch ARTIFACTS_READY
cat artifacts.tgz.sha256
