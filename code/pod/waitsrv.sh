#!/bin/bash
for i in $(seq 1 180); do curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && exit 0; [ $i -gt 12 ] && ! pgrep -f "vllm serve" >/dev/null && exit 1; sleep 5; done; exit 1
